"""Hugging Face model card for one upload folder, rendered from its manifest and parity report.

Every number in the card comes from the report, so the card cannot drift from the measurements.
VERIFIED: tests/test_upload_folder.py checks the numbers against the report and executes the usage snippet.
"""

from .source import LANGUAGES, ModelSpec

USAGE_MARKER = "```python\n"

USAGE = '''from pathlib import Path
import json

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

folder = Path("{repo_name}")  # hf download {repo_id} --local-dir {repo_name}
config = json.loads((folder / "runtime.json").read_text())
tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
tokenizer.no_padding()
tokenizer.no_truncation()
session = ort.InferenceSession(str(folder / "model_q8.onnx"))


def encode(text, kind):
    ids = tokenizer.encode(config["prefixes"][kind] + text).ids
    limit = config["query_length"] if kind == "query" else config["max_document_length"]
    if len(ids) > limit:
        raise ValueError(f"{{kind}} has more than {{limit}} tokens; chunk it first")
    width = limit if kind == "query" else len(ids)
    input_ids = np.full((1, width), config["pad_token_id"], dtype=np.int64)
    input_ids[0, : len(ids)] = ids
    mask = (np.arange(width) < len(ids)).astype(np.int64)[None]
    vectors = session.run(None, {{"input_ids": input_ids, "attention_mask": mask}})[0][0]
    if kind == "query":
        return vectors  # all 32 positions count, including the [MASK] expansion
    return vectors[~np.isin(input_ids[0], config["skip_token_ids"])]


def maxsim(query, document):
    return float((query @ document.T).max(axis=1).sum())


query = encode("Wie hoch ist die Zugspitze?", "query")
passages = ["Die Zugspitze ist 2962 Meter hoch.", "Paris is the capital of France."]
scores = [maxsim(query, encode(passage, "document")) for passage in passages]
print(sorted(zip(scores, passages), reverse=True))
'''

LABELS = {
    "fp32": "FP32 reference export",
    "fp16": "FP16 weights and math",
    "q8": "INT8 weights, FP32 math",
    "int8": "INT8 weights, dynamic INT8 activations",
    "q4": "4-bit weights (MatMulNBits)",
}


def usage(spec: ModelSpec) -> str:
    return USAGE.format(repo_name=spec.repo_name, repo_id=spec.repo_id)


def _ratio(value: float, total: int) -> str:
    return f"{round(value * total)}/{total}"


def render_card(spec: ModelSpec, manifest: dict, report: dict, runtime: dict) -> str:
    results = report["precisions"]
    queries = report["fixture"]["queries"]
    env = report["environment"]
    fp32 = results["fp32"]
    upstream_error = max(
        fp32["vs_upstream"]["max_absolute_error"],
        *(e["max_absolute_error"] for e in fp32["raw_dynamic_shapes"]),
    )
    files = "\n".join(
        f"| {p} | `{m['file']}` | {m['bytes']:,} | {LABELS[p]} | {m['parity']} |"
        for p, m in manifest["models"].items()
    )
    quality = "\n".join(
        f"| {p} | {r['vs_fp32_retrieval_tokens']['mean_token_cosine']:.4f} | "
        f"{r['vs_fp32_retrieval_tokens']['minimum_token_cosine']:.3f} | "
        f"{_ratio(r['top1_agreement'], queries)} | {r['top5_overlap']:.3f} | "
        f"{_ratio(r['top1_relevance_accuracy'], queries)} |"
        for p, r in results.items()
        if p != "fp32"
    )
    shipped = ", ".join(manifest["models"])
    tags = "\n".join(
        f"- {t}"
        for t in ("onnx", "ColBERT", "late-interaction", "multi-vector", "mixture-of-experts", "retrieval", "multilingual")
    )
    languages = "\n".join(f"- {code}" for code in LANGUAGES)
    tolerance = report["exact_tolerance"]
    excluded = "".join(f" {p} is not shipped: {reason}." for p, reason in spec.excluded.items())
    return f"""---
license: apache-2.0
base_model: {spec.model_id}
base_model_relation: quantized
language:
{languages}
library_name: onnx
pipeline_tag: sentence-similarity
tags:
{tags}
---

# {spec.repo_name}

ONNX exports of [{spec.model_id}](https://huggingface.co/{spec.model_id}) at revision
`{spec.revision}`, for CPU inference with ONNX Runtime in Python, Node.js and browsers (WASM).
Each graph turns token IDs into L2-normalized {runtime['dimensions']}-dimensional token vectors. ColBERT MaxSim
over those vectors scores a query against a passage.

## Files

| Precision | File | Bytes | Weights and math | Cross-runtime contract |
|---|---|---:|---|---|
{files}

`manifest.json` lists every file with its size and SHA-256. `runtime.json` holds the tokenization
contract: prefixes, lengths, padding, the punctuation tokens excluded from document scoring.
`parity-report.json` holds every measurement summarized below.

- **exact** graphs give the same token vectors in Python, Node.js and browser ONNX Runtime, within
  {tolerance['token']:g} per value and {tolerance['score']:g} per MaxSim score.
- **quality** graphs depend on runtime kernels, so their vectors differ between runtimes. Compare
  them only with vectors from the same runtime, and key caches by runtime.

## Verification

`moedern-onnx parity` measured these values on {env['platform']} with Python {env['python']},
ONNX Runtime {env['onnxruntime']} and PyTorch {env['torch']}. The FP32 graph matches the upstream
Sentence Transformers model within {upstream_error:.1e} per value, across tokenizer edge cases and
batch shapes up to 3 × 512 tokens.

The other precisions are compared with FP32 on a retrieval fixture of {queries} hand-written queries
over {report['fixture']['documents']} passages in seven languages. The fixture is a regression check,
not a retrieval benchmark.

| Precision | Mean token cosine | Lowest token cosine | Same top-1 as FP32 | Top-5 overlap | Labeled passage ranked first |
|---|---:|---:|---:|---:|---:|
{quality}

## Inputs and outputs

- Inputs `input_ids` and `attention_mask`: int64, shape [batch, sequence], at most 512 positions.
- Output `token_embeddings`: float32, shape [batch, sequence, {runtime['dimensions']}], unit length per token.
- Queries start with `[Q] ` and always use 32 positions: the tokenizer's padding (`[MASK]`) fills the
  rest, and every position counts in MaxSim.
- Documents start with `[D] ` and use at most 512 positions. Padding and punctuation tokens are
  excluded from MaxSim.

## Usage in Python

Needs `onnxruntime`, `tokenizers` and `numpy`.

{USAGE_MARKER}{usage(spec)}```

## Conversion

The exporter re-implements the checkpoint's two attention and mixture-of-experts layers with plain
tensor operations and exports them with `torch.onnx.export` (opset 17, dynamic batch and sequence
axes). The graph includes the projection head and per-token normalization.

Changes from the upstream model, as required by the Apache License 2.0:

1. Every routed expert is computed and the learned router gates weight their outputs. Results match
   upstream, but the graph does not skip unused experts.
2. Positions are capped at 512; upstream's sliding-window attention for longer inputs is not exported.
3. Quantized variants store rounded weights (see the files table); the router stays FP32 because
   small router errors can change which experts run.

## Limitations

- Shipped precisions: {shipped}.{excluded}
- q8 makes the download smaller; it does not make inference faster or reduce memory.
- int8 quantizes activations per batch at run time, so a passage's vectors depend slightly on the
  other passages in its batch.

## License

Apache-2.0, the license of the upstream checkpoint. These graphs are derivatives of
[{spec.model_id}](https://huggingface.co/{spec.model_id}).
"""
