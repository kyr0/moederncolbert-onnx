"""Reference parity, Q8 regression gates, and cross-runtime golden fixtures."""

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .exporting import reference_model, write_json
from .runtime import Encoder, maxsim


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


def verify(source: Path, artifacts: Path, fixture: Path, reports: Path):
    reference = reference_model(source)
    encoders = {precision: Encoder(artifacts, precision) for precision in ("fp32", "q8")}
    corpus = json.loads(fixture.read_text())
    cases = [
        {
            "kind": "query",
            "texts": [
                "",
                "Wie hoch ist die Zugspitze?",
                "Café, Straße, İSTANBUL — déjà vu!",
                "the " * 70,
            ],
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
    # Both edge cases and all retrieval examples become JavaScript golden batches.
    for kind, texts in (
        ("query", [q["text"] for q in corpus["queries"]]),
        ("document", corpus["documents"]),
    ):
        for i in range(0, len(texts), 4):
            cases.append({"kind": kind, "texts": texts[i : i + 4], "truncate": False})
    golden, ref_vectors, fp_vectors, q8_vectors = [], [], [], []
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
        ref_vectors.extend(expected)
        fp_vectors.extend(actual["fp32"])
        q8_vectors.extend(actual["q8"])
        golden.append(
            {
                **case,
                "inputIds": batch.input_ids.tolist(),
                "attentionMask": batch.attention_mask.tolist(),
                "scoreMask": batch.score_mask.tolist(),
                "truncated": batch.truncated,
                "outputs": {p: [m.tolist() for m in matrices] for p, matrices in actual.items()},
            }
        )
    # Exercise raw symbolic axes independently of tokenization and batching order.
    raw_errors = []
    for batch_size, length in ((1, 3), (2, 7), (3, 31), (1, 32), (2, 127), (1, 511), (2, 512)):
        ids = np.full((batch_size, length), 5, dtype=np.int64)
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
        if error > 1e-4:
            raise AssertionError(f"FP32 dynamic-axis parity failed: {raw_errors[-1]}")
    scores = {}
    corpus_vectors = {}
    for precision, encoder in encoders.items():
        queries = encoder.encode([q["text"] for q in corpus["queries"]], "query")
        documents = encoder.encode(corpus["documents"], "document")
        scores[precision] = matrix_scores(queries, documents)
        corpus_vectors[precision] = queries + documents
    rankings = {p: np.argsort(-values, axis=1, kind="stable") for p, values in scores.items()}
    top1 = float(np.mean(rankings["fp32"][:, 0] == rankings["q8"][:, 0]))
    top5 = float(
        np.mean(
            [
                len(set(a[:5]) & set(b[:5])) / 5
                for a, b in zip(rankings["fp32"], rankings["q8"], strict=True)
            ]
        )
    )
    fp32 = similarities(ref_vectors, fp_vectors)
    q8 = similarities(fp_vectors, q8_vectors)
    corpus_q8 = similarities(corpus_vectors["fp32"], corpus_vectors["q8"])
    thresholds = {
        "fp32_max_absolute_error": 1e-4,
        "q8_corpus_mean_token_cosine": 0.98,
        "q8_top1_agreement": 0.95,
        "q8_top5_overlap": 0.90,
    }
    passed = (
        fp32["max_absolute_error"] < thresholds["fp32_max_absolute_error"]
        and corpus_q8["mean_token_cosine"] >= thresholds["q8_corpus_mean_token_cosine"]
        and top1 >= thresholds["q8_top1_agreement"]
        and top5 >= thresholds["q8_top5_overlap"]
    )
    manifest = json.loads((artifacts / "manifest.json").read_text())
    report = {
        "passed": passed,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "graphs": {
            p: {k: v for k, v in info.items() if k != "operators"}
            for p, info in manifest["models"].items()
        },
        "fp32_vs_upstream": fp32,
        "raw_dynamic_shapes": raw_errors,
        "q8_vs_fp32_all_cases": q8,
        "q8_vs_fp32_retrieval_tokens": corpus_q8,
        "retrieval": {
            "description": corpus["description"],
            "queries": len(corpus["queries"]),
            "documents": len(corpus["documents"]),
            "top1_agreement": top1,
            "top5_overlap": top5,
            "max_score_error": float(np.max(np.abs(scores["fp32"] - scores["q8"]))),
            "top1_relevance_accuracy": {
                p: float(
                    np.mean(
                        [
                            rank[0] in query["relevant"]
                            for rank, query in zip(ranking, corpus["queries"], strict=True)
                        ]
                    )
                )
                for p, ranking in rankings.items()
            },
        },
        "thresholds": thresholds,
    }
    write_json(reports / "python-parity.json", report)
    if not passed:
        raise AssertionError(f"Parity/quality gate failed; see {reports / 'python-parity.json'}")
    write_json(
        artifacts / "parity-fixture.json",
        {
            "schema_version": 1,
            "graph_hashes": {p: m["sha256"] for p, m in manifest["models"].items()},
            "cases": golden,
            "corpus": corpus,
            "scores": {p: s.tolist() for p, s in scores.items()},
        },
    )
    return report


def benchmark(artifacts: Path, fixture: Path, repeats: int = 10):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    corpus = json.loads(fixture.read_text())
    manifest = json.loads((artifacts / "manifest.json").read_text())
    output = {
        "model_hashes": {p: m["sha256"] for p, m in manifest["models"].items()},
        "platform": platform.platform(),
        "processor": platform.processor(),
        "threads": 2,
        "repeats": repeats,
        "batch_size": 4,
        "results": {},
    }
    for precision in ("fp32", "q8"):
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
