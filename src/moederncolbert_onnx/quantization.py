"""ONNX variants derived from the FP32 export. `build` returns a new graph and leaves its input untouched."""

import copy
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.transformers.onnx_model import OnnxModel

_TABLE = "table.weight"
_TABLE_Q8 = "row-wise INT8 token table"

DESCRIPTIONS = {
    "fp32": {"weights": "FP32", "activations": "FP32"},
    "fp16": {
        "weights": "FP16",
        "activations": "FP16 inside the graph; FP32 inputs and outputs",
    },
    "q8": {
        "weights": f"per-output-channel signed INT8 linear weights; {_TABLE_Q8}",
        "activations": "FP32 (weights are dequantized; no INT8 compute)",
        "router": "FP32",
    },
    "int8": {
        "weights": f"per-channel signed INT8 linear weights; {_TABLE_Q8}",
        "activations": "dynamic UINT8 at quantized MatMuls, computed per batch at run time",
        "router": "FP32",
    },
    "q4": {
        "weights": f"4-bit symmetric MatMulNBits linear weights, block size 32; {_TABLE_Q8}",
        "activations": "FP32",
        "router": "FP32",
    },
}


def router_nodes(model: onnx.ModelProto) -> list[str]:
    # HYPOTHESIS: small errors in router logits can flip the discrete top-k expert selection;
    # falsifier: quantize the routers too and compare the selected experts with FP32.
    return [node.name for node in model.graph.node if "/router/" in node.name]


def _quantize_table(model: onnx.ModelProto) -> None:
    """Symmetric per-row INT8 token table. Gather precedes dequantization, so the full
    floating-point table is never materialized at inference time."""
    table = next(i for i in model.graph.initializer if i.name == _TABLE)
    weights = numpy_helper.to_array(table)
    scales = np.maximum(np.max(np.abs(weights), axis=1, keepdims=True) / 127.0, 1e-12)
    quantized = np.clip(np.rint(weights / scales), -127, 127).astype(np.int8)
    gather = next(n for n in model.graph.node if n.op_type == "Gather" and n.input[0] == _TABLE)
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


def _weight_only_int8(model: onnx.ModelProto) -> None:
    """Per-output-channel INT8 weights with FP32 activations and math."""
    excluded = set(router_nodes(model))
    names = {
        n.input[1] for n in model.graph.node if n.op_type == "MatMul" and n.name not in excluded
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
        # Explicit Cast/Mul prevents ORT's DequantizeLinear-MatMul rewrite to MatMulNBits,
        # whose native approximation differed from WASM. The constants fold into FP32 weights.
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


def _dynamic_int8(model: onnx.ModelProto) -> onnx.ModelProto:
    with tempfile.TemporaryDirectory() as directory:
        source, target = Path(directory) / "in.onnx", Path(directory) / "out.onnx"
        onnx.save(model, source)
        quantize_dynamic(
            str(source),
            str(target),
            op_types_to_quantize=["MatMul"],
            per_channel=True,
            weight_type=QuantType.QInt8,
            nodes_to_exclude=router_nodes(model),
            extra_options={"MatMulConstBOnly": True},
        )
        return onnx.load(target)


def _int4(model: onnx.ModelProto) -> onnx.ModelProto:
    quantizer = MatMulNBitsQuantizer(
        model, block_size=32, is_symmetric=True, nodes_to_exclude=router_nodes(model)
    )
    quantizer.process()
    return quantizer.model.model


def build(precision: str, fp32: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(fp32)
    if precision == "fp16":
        # VERIFIED: the converter keeps blocked ops such as TopK in FP32 and appends their Casts at
        # the end of the node list, and onnx.checker rejected that graph as not topologically sorted.
        converted = OnnxModel(convert_float_to_float16(model, keep_io_types=True))
        converted.topological_sort()
        return converted.model
    if precision not in ("q8", "int8", "q4"):
        raise ValueError(f"No variant builder for {precision!r}")
    _quantize_table(model)
    if precision == "q8":
        _weight_only_int8(model)
        return model
    return _dynamic_int8(model) if precision == "int8" else _int4(model)
