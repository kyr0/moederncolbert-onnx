"""Upstream parity, per-precision gates, cross-runtime golden fixtures and benchmarks."""

import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .card import render_card
from .contracts import CONTRACTS, EXACT_TOLERANCE, FP32_UPSTREAM_TOLERANCE
from .exporting import reference_model, write_json
from .runtime import Encoder, maxsim
from .source import ModelSpec

EDGE_CASES = [
    {
        "kind": "query",
        "texts": ["", "Wie hoch ist die Zugspitze?", "Café, Straße, İSTANBUL — déjà vu!", "the " * 70],
        "truncate": True,
    },
    {
        "kind": "document",
        "texts": [
            "",
            "Café, Straße, İSTANBUL — déjà vu!",
            "\"Hello\", ''world''! 😀 中文 русский",
            "One\n\ttwo   three.",
        ],
        "truncate": False,
    },
    {"kind": "document", "texts": ["the " * 508, "the " * 509], "truncate": False},
    {"kind": "document", "texts": ["the " * 530, "short"], "truncate": True},
]
# (batch, sequence). VERIFIED: for the 27M checkpoint, batches above 1,220 tokens take upstream's grouped
# MoE path instead of its dense one, so (3, 512) checks the graph against both upstream paths.
RAW_SHAPES = ((1, 3), (2, 7), (3, 31), (1, 32), (2, 127), (1, 511), (2, 512), (3, 512))


def matrix_scores(queries, documents):
    return np.array([[maxsim(q, d) for d in documents] for q in queries])


def similarities(reference, actual):
    left, right = np.concatenate(reference), np.concatenate(actual)
    cosine = (left * right).sum(-1) / (
        np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    ).clip(1e-12)
    return {
        "max_absolute_error": float(np.max(np.abs(left - right))),
        "mean_token_cosine": float(cosine.mean()),
        "minimum_token_cosine": float(cosine.min()),
    }


def rankings(scores):
    return np.argsort(-scores, axis=1, kind="stable")


def agreement(reference_scores, scores):
    a, b = rankings(reference_scores), rankings(scores)
    return {
        "top1_agreement": float(np.mean(a[:, 0] == b[:, 0])),
        "top5_overlap": float(
            np.mean([len(set(x[:5]) & set(y[:5])) / 5 for x, y in zip(a, b, strict=True)])
        ),
        "max_score_error": float(np.max(np.abs(reference_scores - scores))),
    }


def _environment():
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in ("onnxruntime", "torch")},
    }


def verify(
    spec: ModelSpec,
    source: Path,
    artifacts: Path,
    fixture: Path,
    reports: Path,
    parity_fixture: Path,
) -> dict:
    reference = reference_model(spec, source)
    manifest = json.loads((artifacts / "manifest.json").read_text())
    missing = [p for p in spec.precisions if p not in manifest["models"]]
    if missing:
        raise ValueError(f"Missing graphs for {missing}; run `moedern-onnx quantize` first")
    precisions = list(spec.precisions)
    encoders = {p: Encoder(artifacts, p) for p in precisions}
    exact = [p for p in precisions if CONTRACTS[p]["parity"] == "exact"]
    corpus = json.loads(fixture.read_text())
    cases = list(EDGE_CASES)
    # Edge cases and all retrieval examples become JavaScript golden batches.
    for kind, texts in (
        ("query", [q["text"] for q in corpus["queries"]]),
        ("document", corpus["documents"]),
    ):
        for i in range(0, len(texts), 4):
            cases.append({"kind": kind, "texts": texts[i : i + 4], "truncate": False})
    golden, upstream_vectors = [], []
    outputs = {p: [] for p in precisions}
    for case in cases:
        texts, kind = case["texts"], case["kind"]
        batch = encoders["fp32"].preprocessor.prepare(texts, kind, truncate=case["truncate"])
        upstream = reference.preprocess(texts, task=kind, prompt=reference.prompts[kind])
        for key in ("input_ids", "attention_mask"):
            np.testing.assert_array_equal(getattr(batch, key), upstream[key].numpy())
        with torch.inference_mode():
            full = reference(upstream, task=kind)["token_embeddings"].numpy()
        expected = [full[i, mask].copy() for i, mask in enumerate(batch.score_mask)]
        # Also check public upstream masking/normalization, not only its backbone.
        method = reference.encode_query if kind == "query" else reference.encode_document
        public = method(texts, batch_size=len(texts), show_progress_bar=False)
        for a, b in zip(expected, public, strict=True):
            np.testing.assert_allclose(a, np.asarray(b), atol=1e-5, rtol=1e-5)
        actual = {
            p: e.encode(texts, kind, batch_size=len(texts), truncate=case["truncate"])
            for p, e in encoders.items()
        }
        upstream_vectors.extend(expected)
        for p in precisions:
            outputs[p].extend(actual[p])
        golden.append(
            {
                **case,
                "inputIds": batch.input_ids.tolist(),
                "attentionMask": batch.attention_mask.tolist(),
                "scoreMask": batch.score_mask.tolist(),
                "truncated": batch.truncated,
                # Quality precisions have no cross-runtime golden values by contract.
                "outputs": {p: [m.tolist() for m in actual[p]] for p in exact},
            }
        )
    # Exercise raw symbolic axes independently of tokenization and batching order.
    raw_errors = []
    for batch_size, length in RAW_SHAPES:
        ids = np.full((batch_size, length), 5, dtype=np.int64)
        ids[:, 2:-1] = np.random.default_rng(length).integers(5, 29999, size=(batch_size, length - 3))
        ids[:, 0], ids[:, 1], ids[:, -1] = 2, 30001, 3
        attention = np.ones_like(ids)
        if batch_size > 1:
            attention[-1, -2:] = 0
            ids[-1, -2:] = 4
        with torch.inference_mode():
            values = reference(
                {"input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(attention)},
                task="document",
            )["token_embeddings"].numpy()
        actual = encoders["fp32"].session.run(
            None, {"input_ids": ids, "attention_mask": attention}
        )[0]
        error = float(np.max(np.abs(values - actual)))
        raw_errors.append({"batch": batch_size, "sequence": length, "max_absolute_error": error})
    scores, corpus_vectors = {}, {}
    for precision, encoder in encoders.items():
        queries = encoder.encode([q["text"] for q in corpus["queries"]], "query")
        documents = encoder.encode(corpus["documents"], "document")
        scores[precision] = matrix_scores(queries, documents)
        corpus_vectors[precision] = queries + documents
    results = {}
    for precision in precisions:
        contract = CONTRACTS[precision]
        graph = manifest["models"][precision]
        result = {
            "contract": contract,
            "graph": {k: graph[k] for k in ("file", "bytes", "sha256", "opsets", "quantization")},
            "top1_relevance_accuracy": float(
                np.mean(
                    [
                        rank[0] in query["relevant"]
                        for rank, query in zip(rankings(scores[precision]), corpus["queries"], strict=True)
                    ]
                )
            ),
        }
        if precision == "fp32":
            upstream = similarities(upstream_vectors, outputs["fp32"])
            worst = max(upstream["max_absolute_error"], *(e["max_absolute_error"] for e in raw_errors))
            result |= {
                "vs_upstream": upstream,
                "raw_dynamic_shapes": raw_errors,
                "passed": worst < FP32_UPSTREAM_TOLERANCE,
            }
        else:
            tokens = similarities(corpus_vectors["fp32"], corpus_vectors[precision])
            agree = agreement(scores["fp32"], scores[precision])
            result |= {
                "vs_fp32_all_cases": similarities(outputs["fp32"], outputs[precision]),
                "vs_fp32_retrieval_tokens": tokens,
                **agree,
                "passed": tokens["mean_token_cosine"] >= contract["mean_token_cosine"]
                and agree["top1_agreement"] >= contract["top1_agreement"]
                and agree["top5_overlap"] >= contract["top5_overlap"],
            }
        results[precision] = result
    report = {
        "passed": all(r["passed"] for r in results.values()),
        "model_id": spec.model_id,
        "revision": spec.revision,
        "repo_id": spec.repo_id,
        "environment": _environment(),
        "fixture": {
            "description": corpus["description"],
            "queries": len(corpus["queries"]),
            "documents": len(corpus["documents"]),
        },
        "exact_tolerance": EXACT_TOLERANCE,
        "precisions": results,
    }
    write_json(reports / "python-parity.json", report)
    if not report["passed"]:
        failed = [p for p, r in results.items() if not r["passed"]]
        raise AssertionError(f"Parity/quality gate failed for {failed}; see {reports / 'python-parity.json'}")
    write_json(artifacts / "parity-report.json", report)
    runtime = json.loads((artifacts / "runtime.json").read_text())
    (artifacts / "README.md").write_text(render_card(spec, manifest, report, runtime))
    write_json(
        parity_fixture,
        {
            "schema_version": 2,
            "model": {"slug": spec.slug, "model_id": spec.model_id, "revision": spec.revision},
            "precisions": precisions,
            "contracts": {p: CONTRACTS[p] for p in precisions},
            "exact_tolerance": EXACT_TOLERANCE,
            "graph_hashes": {p: manifest["models"][p]["sha256"] for p in precisions},
            "cases": golden,
            "corpus": corpus,
            "scores": {p: s.tolist() for p, s in scores.items()},
        },
    )
    return report


def reference_fixture(artifacts: Path, fixture: Path, *, queries: int = 6, documents: int = 8) -> dict:
    """Small Python-side scores for consumers (defuss-vectorsearch) to check their runtimes against."""
    manifest = json.loads((artifacts / "manifest.json").read_text())
    corpus = json.loads(fixture.read_text())
    docs = corpus["documents"][:documents]
    picked = [q for q in corpus["queries"] if all(i < documents for i in q["relevant"])][:queries]
    output = {
        "schema_version": 1,
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "repo_id": manifest["repo_id"],
        "queries": [q["text"] for q in picked],
        "relevant": [q["relevant"] for q in picked],
        "documents": docs,
        "exact_tolerance": EXACT_TOLERANCE,
        "precisions": {},
    }
    for precision, info in manifest["models"].items():
        encoder = Encoder(artifacts, precision)
        q = encoder.encode(output["queries"], "query")
        d = encoder.encode(docs, "document")
        output["precisions"][precision] = {
            "sha256": info["sha256"],
            "contract": CONTRACTS[precision],
            "document_token_counts": [len(m) for m in d],
            "scores": matrix_scores(q, d).tolist(),
        }
    return output


def benchmark(artifacts: Path, fixture: Path, repeats: int = 10):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    corpus = json.loads(fixture.read_text())
    manifest = json.loads((artifacts / "manifest.json").read_text())
    output = {
        "model_id": manifest["model_id"],
        "model_hashes": {p: m["sha256"] for p, m in manifest["models"].items()},
        "environment": _environment(),
        "processor": platform.processor(),
        "threads": 2,
        "repeats": repeats,
        "batch_size": 4,
        "results": {},
    }
    for precision in manifest["models"]:
        start = time.perf_counter()
        encoder = Encoder(artifacts, precision)
        load_ms = (time.perf_counter() - start) * 1000
        text = corpus["queries"][0]["text"]
        docs = encoder.encode(corpus["documents"], "document")
        query = encoder.encode([text], "query")[0]
        jobs = {
            "query_encode": lambda e=encoder, t=text: e.encode([t], "query"),
            "encode_24_documents": lambda e=encoder: e.encode(corpus["documents"], "document"),
            "maxsim_24_cached": lambda q=query, ds=docs: [maxsim(q, d) for d in ds],
            "rerank_24_uncached": lambda e=encoder, t=text: e.rerank(t, corpus["documents"]),
        }
        results = {"load_ms": load_ms}
        for name, job in jobs.items():
            job()
            times = []
            for _ in range(repeats):
                start = time.perf_counter()
                job()
                times.append((time.perf_counter() - start) * 1000)
            results[name] = {
                "median_ms": float(np.median(times)),
                "p95_ms": float(np.percentile(times, 95)),
            }
        output["results"][precision] = results
    return output
