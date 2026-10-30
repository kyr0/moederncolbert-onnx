"""Per-precision verification contracts, the one source for exporter, validation, model card and JS harness.

exact: every runtime reproduces the Python ONNX Runtime token vectors within EXACT_TOLERANCE.
quality: each runtime is compared with its own FP32 output against the quality gate instead.
VERIFIED: fp16 and dynamic int8 kernels differ between runtimes, by up to 0.15 per value between
Python ONNX Runtime 1.31.1, onnxruntime-node 1.21.0 and ONNX Runtime Web 1.30.0 (probe 2026-10-30).
"""

EXACT_TOLERANCE = {"token": 1e-4, "score": 1e-3}
FP32_UPSTREAM_TOLERANCE = 1e-4
QUALITY_GATE = {"mean_token_cosine": 0.98, "top1_agreement": 0.95, "top5_overlap": 0.90}

CONTRACTS = {
    "fp32": {"parity": "exact"},
    "fp16": {"parity": "quality", **QUALITY_GATE, "mean_token_cosine": 0.999},
    "q8": {"parity": "exact", **QUALITY_GATE},
    "int8": {"parity": "quality", **QUALITY_GATE},
    "q4": {"parity": "exact", **QUALITY_GATE},
}
