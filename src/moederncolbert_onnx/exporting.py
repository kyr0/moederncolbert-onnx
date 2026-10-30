"""FP32 export and auditable Q8 conversion; checkpoints stay pinned."""

import importlib.metadata
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

from .portable import PortableEncoder
from .source import MODEL_ID, REVISION, sha256, verify_source


def reference_model(source: Path):
    from sentence_transformers import MultiVectorEncoder

    verify_source(source)
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
        "opsets": {op.domain: op.version for op in model.opset_import},
    }


def update_manifest(artifacts: Path, precision: str):
    path = artifacts / "manifest.json"
    data = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "schema_version": 1,
            "model_id": MODEL_ID,
            "revision": REVISION,
            "files": {},
            "models": {},
        }
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "runtime.json"):
        data["files"][name] = {
            "bytes": (artifacts / name).stat().st_size,
            "sha256": sha256(artifacts / name),
        }
    name = f"model_{precision}.onnx"
    info = graph_info(artifacts / name)
    data["files"][name] = {"bytes": info["bytes"], "sha256": info["sha256"]}
    data["models"][precision] = {"file": name, **info}
    data["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "sentence-transformers", "onnx", "onnxruntime")
    }
    write_json(path, data)
    return data


def export_fp32(source: Path, artifacts: Path):
    reference = reference_model(source)
    artifacts.mkdir(parents=True, exist_ok=True)
    # MVE promotes legacy [Q]/[D] strings to non-normalized special tokens.
    # Save this effective tokenizer, not the original unmodified tokenizer.json.
    reference.tokenizer.save_pretrained(artifacts)
    punctuation = reference[2]._skiplist_ids.tolist()
    cfg = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "dimensions": 128,
        "query_length": 32,
        "max_document_length": 512,
        "prefixes": reference.prompts,
        "pad_token_id": reference.tokenizer.pad_token_id,
        "mask_token_id": reference.tokenizer.mask_token_id,
        "sep_token_id": reference.tokenizer.sep_token_id,
        "query_marker_id": 30000,
        "document_marker_id": 30001,
        "skip_token_ids": sorted(set(punctuation)),
        "query_expansion": True,
        "query_backbone_attends_to_expansion": True,
        "document_scoring": "attention mask minus punctuation; includes special tokens",
        "normalization": "per-token L2; epsilon 1e-12",
        "similarity": "sum_maxsim",
        "languages": ["de", "en", "es", "nl", "fr", "it", "pt"],
    }
    write_json(artifacts / "runtime.json", cfg)
    shutil.copyfile(source / "LICENSE", artifacts / "MODEL-LICENSE.txt")
    encoder = PortableEncoder(source)
    ids = torch.tensor([[2, 30001, 101, 102, 3], [2, 30001, 55, 3, 4]], dtype=torch.int64)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=torch.int64)
    target = artifacts / "model_fp32.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            encoder,
            (ids, mask),
            str(target),
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
    # Exporting a new FP32 model invalidates previously generated Q8 graphs/reports.
    (artifacts / "model_q8.onnx").unlink(missing_ok=True)
    (artifacts / "manifest.json").unlink(missing_ok=True)
    (artifacts / "parity-fixture.json").unlink(missing_ok=True)
    return update_manifest(artifacts, "fp32")


def quantize_q8(artifacts: Path, mode: str = "weight-only"):
    if mode not in ("weight-only", "dynamic"):
        raise ValueError("Q8 mode must be weight-only or dynamic")
    (artifacts / "parity-fixture.json").unlink(missing_ok=True)
    source = artifacts / "model_fp32.onnx"
    model = onnx.load(source)
    # Symmetric per-row INT8 factorized token table. Gather before dequantization
    # avoids materializing the full floating-point table at inference time.
    table = next(i for i in model.graph.initializer if i.name == "table.weight")
    weights = numpy_helper.to_array(table)
    scales = np.maximum(np.max(np.abs(weights), axis=1, keepdims=True) / 127.0, 1e-12)
    quantized = np.clip(np.rint(weights / scales), -127, 127).astype(np.int8)
    gather = next(n for n in model.graph.node if n.op_type == "Gather" and n.input[0] == table.name)
    original_output = gather.output[0]
    gather.input[0] = "embedding_table_q8"
    gather.output[0] = "gathered_embedding_q8"
    nodes = [
        onnx.helper.make_node(
            "Gather",
            ["embedding_table_scales", gather.input[1]],
            ["gathered_embedding_scales"],
            name="embedding_scale_gather",
            axis=0,
        ),
        onnx.helper.make_node(
            "Cast",
            ["gathered_embedding_q8"],
            ["gathered_embedding_float"],
            name="embedding_cast",
            to=onnx.TensorProto.FLOAT,
        ),
        onnx.helper.make_node(
            "Mul",
            ["gathered_embedding_float", "gathered_embedding_scales"],
            [original_output],
            name="embedding_dequantize",
        ),
    ]
    position = list(model.graph.node).index(gather) + 1
    for offset, node in enumerate(nodes):
        model.graph.node.insert(position + offset, node)
    model.graph.initializer.remove(table)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(quantized, "embedding_table_q8"),
            numpy_helper.from_array(scales.astype(np.float32), "embedding_table_scales"),
        ]
    )
    # Exclude router MatMuls: small errors can change discrete top-2 selection.
    exclusions = [node.name for node in model.graph.node if "/router/" in node.name]
    if mode == "dynamic":
        intermediate = artifacts / ".embedding_q8.onnx"
        onnx.save(model, intermediate)
        try:
            quantize_dynamic(
                str(intermediate),
                str(artifacts / "model_q8.onnx"),
                op_types_to_quantize=["MatMul"],
                per_channel=True,
                weight_type=QuantType.QInt8,
                nodes_to_exclude=exclusions,
                extra_options={"MatMulConstBOnly": True},
            )
        finally:
            intermediate.unlink(missing_ok=True)
    else:
        # Storage Q8: per-output-channel INT8 weights, FP32 activations and math.
        # ORT may constant-fold these dequantizations. No INT8 speed/RAM claim.
        names = {
            n.input[1]
            for n in model.graph.node
            if n.op_type == "MatMul" and n.name not in exclusions
        }
        dequantizers = []
        for initializer in list(model.graph.initializer):
            if initializer.name not in names:
                continue
            values = numpy_helper.to_array(initializer)
            if values.ndim != 2:
                raise ValueError(f"Unexpected linear weight shape: {initializer.name}")
            scale = np.maximum(np.max(np.abs(values), axis=0) / 127.0, 1e-12)
            packed = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
            name = initializer.name
            model.graph.initializer.remove(initializer)
            model.graph.initializer.extend(
                [
                    numpy_helper.from_array(packed, name + "_q8"),
                    numpy_helper.from_array(scale.astype(np.float32), name + "_scale"),
                ]
            )
            # Explicit Cast/Mul prevents ORT's DQ-MatMul rewrite to MatMulNBits,
            # whose native approximation differs from WASM. These constants
            # fold into FP32 weights and keep arithmetic portable across engines.
            dequantizers.extend(
                [
                    onnx.helper.make_node(
                        "Cast",
                        [name + "_q8"],
                        [name + "_float"],
                        name=name + "_cast",
                        to=onnx.TensorProto.FLOAT,
                    ),
                    onnx.helper.make_node(
                        "Mul", [name + "_float", name + "_scale"], [name], name=name + "_dequantize"
                    ),
                ]
            )
        for node in reversed(dequantizers):
            model.graph.node.insert(0, node)
        onnx.checker.check_model(model)
        onnx.save(model, artifacts / "model_q8.onnx")
    result = update_manifest(artifacts, "q8")
    result["models"]["q8"]["quantization"] = {
        "mode": mode,
        "weights": "per-channel signed INT8 linear weights; row-wise INT8 token table",
        "activations": (
            "FP32"
            if mode == "weight-only"
            else "dynamic UINT8 at quantized linear MatMuls; float elsewhere"
        ),
        "router": "FP32",
        "excluded_nodes": exclusions,
        "compute": "all eight experts evaluated; learned top-2 gates preserved",
    }
    write_json(artifacts / "manifest.json", result)
    return result
