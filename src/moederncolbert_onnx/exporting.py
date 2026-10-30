"""FP32 export, variant builds and the upload folder's manifest; checkpoints stay pinned.

VERIFIED: the FP32 graphs match upstream within 1.0e-6 (1.5M) and 8.6e-7 (27M), reports/<model>/python-parity.json.
"""

import importlib.metadata
import json
import shutil
from collections import Counter
from pathlib import Path

import onnx
import torch
from tokenizers import Tokenizer

from .contracts import CONTRACTS
from .portable import PortableEncoder
from .quantization import DESCRIPTIONS, build
from .source import LANGUAGES, ModelSpec, sha256, verify_source

SUPPORT_FILES = ("tokenizer.json", "tokenizer_config.json", "runtime.json", "LICENSE")
# Written by later steps; a new FP32 graph or new variants make them stale.
DERIVED_FILES = ("parity-report.json", "README.md")


def reference_model(spec: ModelSpec, source: Path):
    from sentence_transformers import MultiVectorEncoder

    verify_source(spec, source)
    torch.set_num_threads(2)
    # Only pinned, checksum-verified local custom code is loaded; no Hub code refresh.
    return MultiVectorEncoder(
        str(source.resolve()), trust_remote_code=True, local_files_only=True, device="cpu"
    )


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def graph_info(path: Path) -> dict:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "operators": dict(Counter(n.op_type for n in model.graph.node)),
        "opsets": {op.domain or "ai.onnx": op.version for op in model.opset_import},
    }


def write_manifest(spec: ModelSpec, artifacts: Path) -> dict:
    """Describe exactly the files present for the spec's precisions."""
    files = {
        name: {"bytes": (artifacts / name).stat().st_size, "sha256": sha256(artifacts / name)}
        for name in SUPPORT_FILES
    }
    models = {}
    for precision in spec.precisions:
        path = artifacts / f"model_{precision}.onnx"
        if not path.exists():
            continue
        info = graph_info(path)
        files[path.name] = {"bytes": info["bytes"], "sha256": info["sha256"]}
        models[precision] = {
            "file": path.name,
            **info,
            "parity": CONTRACTS[precision]["parity"],
            "quantization": DESCRIPTIONS[precision],
        }
    manifest = {
        "schema_version": 1,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "repo_id": spec.repo_id,
        "files": files,
        "models": models,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "sentence-transformers", "onnx", "onnxruntime")
        },
    }
    write_json(artifacts / "manifest.json", manifest)
    return manifest


def _marker_ids(artifacts: Path, prefixes: dict) -> dict:
    tokenizer = Tokenizer.from_file(str(artifacts / "tokenizer.json"))
    # Position 1 follows [CLS]; the graph reads the query marker from exactly there.
    return {kind: tokenizer.encode(prefix + "x").ids[1] for kind, prefix in prefixes.items()}


def export_fp32(spec: ModelSpec, source: Path, artifacts: Path) -> dict:
    reference = reference_model(spec, source)
    artifacts.mkdir(parents=True, exist_ok=True)
    for stale in [*artifacts.glob("model_*.onnx"), *(artifacts / n for n in DERIVED_FILES)]:
        stale.unlink(missing_ok=True)
    # MVE promotes legacy [Q]/[D] strings to non-normalized special tokens.
    # Save this effective tokenizer, not the original unmodified tokenizer.json.
    reference.tokenizer.save_pretrained(artifacts)
    encoder = PortableEncoder(source)
    markers = _marker_ids(artifacts, reference.prompts)
    if markers["query"] != encoder.query_marker_id:
        raise ValueError(f"Tokenizer query marker {markers['query']} != {encoder.query_marker_id}")
    dense = json.loads((source / "1_Dense/config.json").read_text())
    write_json(
        artifacts / "runtime.json",
        {
            "schema_version": 1,
            "model_id": spec.model_id,
            "revision": spec.revision,
            "dimensions": dense["out_features"],
            "query_length": 32,
            "max_document_length": 512,
            "prefixes": reference.prompts,
            "pad_token_id": reference.tokenizer.pad_token_id,
            "mask_token_id": reference.tokenizer.mask_token_id,
            "sep_token_id": reference.tokenizer.sep_token_id,
            "query_marker_id": markers["query"],
            "document_marker_id": markers["document"],
            "skip_token_ids": sorted(set(reference[2]._skiplist_ids.tolist())),
            "query_expansion": True,
            "query_backbone_attends_to_expansion": True,
            "document_scoring": "attention mask minus punctuation; includes special tokens",
            "normalization": "per-token L2; epsilon 1e-12",
            "similarity": "sum_maxsim",
            "languages": list(LANGUAGES),
        },
    )
    shutil.copyfile(source / "LICENSE", artifacts / "LICENSE")
    ids = torch.tensor([[2, 30001, 101, 102, 3], [2, 30001, 55, 3, 4]], dtype=torch.int64)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=torch.int64)
    with torch.inference_mode():
        torch.onnx.export(
            encoder,
            (ids, mask),
            str(artifacts / "model_fp32.onnx"),
            input_names=["input_ids", "attention_mask"],
            output_names=["token_embeddings"],
            opset_version=17,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_embeddings": {0: "batch", 1: "sequence"},
            },
            dynamo=False,
        )
    return write_manifest(spec, artifacts)


def build_variants(spec: ModelSpec, artifacts: Path) -> dict:
    fp32 = onnx.load(artifacts / "model_fp32.onnx")
    for name in DERIVED_FILES:
        (artifacts / name).unlink(missing_ok=True)
    for precision in spec.precisions[1:]:
        onnx.save(build(precision, fp32), artifacts / f"model_{precision}.onnx")
    return write_manifest(spec, artifacts)
