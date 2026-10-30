# MoEdernColBERT: Python → ONNX → JavaScript

A complete, standalone proof of operation for
[VAGOsolutions/SauerkrautLM-MoEdernColBERT-1.5M](https://huggingface.co/VAGOsolutions/SauerkrautLM-MoEdernColBERT-1.5M):
FP32 export with upstream parity, portable Q8 conversion, Python/Node/browser
inference, and a local interactive reranking demo.

The ZIP includes the pinned upstream checkpoint, both converted ONNX models,
effective tokenizer, compiled JavaScript library, source, lockfiles, tests,
golden fixtures, and verification reports. No Hugging Face account or API key
is required. Package dependencies are installed separately.

## Quick start on macOS

Target: **Apple Silicon, macOS 14+, native arm64 terminal**, Python 3.11 managed
by uv, Bun 1.4.4, and Node.js 22.12+ (Node 24 recommended). The locked ONNX Runtime
Python wheel requires macOS 14. The exporter uses CPU, not MPS/CoreML/CUDA.
Intel Macs are not supported by the locked Python export environment; the
browser implementation is independent of Python once artifacts exist.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/),
[Bun](https://bun.sh/docs/installation), and [Node.js](https://nodejs.org/en/download),
then unzip and run from the project directory:

```sh
make setup
make demo-node
make demo
```

Open the localhost URL printed by Vite (normally `http://127.0.0.1:5173`).
Choose FP32 or Q8, edit the query/passages, and click **Rank passages**. Switching
precision loads the other graph. Repeating a query reuses cached document
vectors. The demo serves all assets locally, including the WASM runtime.

`make setup` installs frozen dependencies and builds the JS library. It sets
`ONNXRUNTIME_NODE_INSTALL=skip` to avoid the optional CUDA download; CPU native
binaries are bundled by onnxruntime-node. Bun manages packages; the native
example and tests execute under Node.js.

## Rebuild and verify everything

```sh
make all
make browser-test
make benchmark
```

`make all` runs setup, verifies/downloads the pinned source, exports FP32,
creates weight-only Q8, compares with upstream PyTorch, regenerates JS golden
fixtures, verifies Node, runs Python/JS tests and type/lint checks, and builds
the web demo. `make browser-test` installs Playwright Chromium and runs real
WASM inference plus both precision choices in the UI. A failed gate exits
nonzero. `make benchmark` measures the current machine, with warmup, median,
and p95; it does not reuse the included Linux timing results.

Individual commands:

```sh
uv run --frozen moedern-onnx download
uv run --frozen moedern-onnx export
uv run --frozen moedern-onnx quantize --q8-mode weight-only
uv run --frozen moedern-onnx verify
uv run --frozen moedern-onnx demo --precision q8 --query 'Wie hoch ist die Zugspitze?'
bun run build
bun run test:node
node examples/node.mjs fp32 'What is the capital of France?'
```

The Python CLI accepts `--source`, `--artifacts`, `--fixture`, and `--reports`.
The default paths are relative to the current directory. Run `moedern-onnx
--help` through `uv run` for options. Export/quantization invalidate the golden
fixture; run `verify` before rebuilding the browser demo.

## What was actually verified

The included reports were generated on **Linux x86-64**, Python 3.11.17,
PyTorch 2.6.0 CPU, Python ONNX Runtime 1.31.1, Node 24.22.1 with ONNX Runtime
1.30.0, and Chromium 140 using ONNX Runtime Web 1.30.0. This environment did
not provide macOS or Safari; run the supplied checks on your Mac before relying
on platform-specific parity or performance.

| Check | Observed result |
|---|---:|
| FP32 ONNX file | 6,294,481 bytes / 6.29 MB |
| Q8 ONNX file | 2,016,962 bytes / 2.02 MB |
| Size reduction | ~68% |
| FP32 vs upstream maximum absolute token error | 8.87 × 10⁻⁷ |
| Node vs Python ONNX maximum token error, both precisions | 0 |
| Browser vs Python maximum token error, FP32 | 6.71 × 10⁻⁷ |
| Browser vs Python maximum token error, Q8 | 9.15 × 10⁻⁷ |
| Cross-runtime comparisons | 379,264 values per precision, 16 batches |
| Q8 vs FP32 mean token cosine on retrieval fixture | 0.9971 |
| Q8 vs FP32 top-1 agreement | 24 / 24 |
| Q8 vs FP32 average top-5 overlap | 97.5% |

The fixture includes German, English, Spanish, French, Italian, Portuguese and
Dutch examples, explicit relevance labels, Unicode/tokenizer edge cases, empty
text, punctuation, query expansion and truncation. Raw dynamic axes are checked
at batch sizes 1–3 and sequence lengths 3, 7, 31, 32, 127, 511 and 512. The
24-query retrieval set is deliberately small and hand-authored. Its perfect
top-1 relevance accuracy is a smoke-test result, **not a retrieval benchmark**
or proof of improvement over Winzling.

FP32 parity and Q8 quality are different checks. Q8 is lossy: on the retrieval
fixture its minimum token cosine is approximately 0.869 and its maximum score
change is approximately 0.290. Exact token equality between FP32 and Q8 is not
claimed. All scores and thresholds are recorded in `reports/python-parity.json`;
the Node and browser reports bind their results to the exact graph SHA-256s.

## What Q8 means here

The default is **signed INT8 weight storage with FP32 activations and math**:
per-output-channel linear weights, per-row token embedding weights, and FP32
router weights/norms. The graph explicitly casts and scales quantized linear
weights. This avoids ONNX Runtime's `DequantizeLinear → MatMulNBits` rewrite,
which caused native/WASM discrepancies during development. Gather precedes
dequantization for the factorized embedding table.

This makes the model download smaller. It does **not** promise INT8 compute,
reduced peak RAM, or faster inference. Constant folding can materialize linear
weights as FP32 in memory. Output vectors are always FP32. The tokenizer is
another ~1.9 MB, and the browser WASM runtime is ~14.2 MB uncompressed; model
file size is not the total application download size.

An optional **experimental dynamic INT8** conversion is also implemented:

```sh
# Replaces the Q8 artifact and invalidates its fixture:
uv run --frozen moedern-onnx quantize --q8-mode dynamic
make verify
make browser-test
make benchmark

# Restore the portable default:
make quantize
make verify
```

Dynamic mode quantizes MatMul activations to UINT8 at runtime, uses per-channel
INT8 weights, and keeps routers FP32. It can change routing and can depend on
batch composition and runtime kernels. The shipped reports/models certify
the **weight-only default**; dynamic mode must pass the checks in your target
environment before use. The cross-runtime parity thresholds are not relaxed
for this mode.

## JavaScript API

```js
import { createNodeEncoder } from './dist/node.mjs';
import { rankEncoded } from './dist/index.mjs';

const encoder = await createNodeEncoder('./artifacts', { precision: 'q8' });
try {
  const passages = [
    'Die Zugspitze ist 2962 Meter hoch.',
    'Paris is the capital of France.',
  ];
  const documents = await encoder.encodeDocuments(passages); // cache these
  const [query] = await encoder.encodeQueries(['Wie hoch ist die Zugspitze?']);
  console.log(rankEncoded(query, documents));
  console.log(await encoder.rerank('Wie hoch ist die Zugspitze?', passages, { topK: 1 }));
} finally {
  await encoder.dispose();
}
```

Browser entry: `createBrowserEncoder('/models/', { precision: 'fp32',
wasmPath: '/ort/' })` from `dist/browser.mjs`. Both entries share tokenization,
masking, batching, normalized token output, MaxSim and stable sorting. Artifact
hashes are checked before loading. Each token matrix carries a model/precision
fingerprint; `maxSim` rejects incompatible matrices. `encodeQueries` and
`encodeDocuments` accept `{ batchSize, truncate, documentLength }`.

Queries always use 32 positions. Documents use up to 512 positions. Inputs
exceeding these limits throw unless `truncate: true` is explicitly requested;
each result reports whether it was truncated. Document padding and punctuation
are removed before scoring, while query expansion tokens remain. See
`docs/defuss-integration.md` for the full contract and cache integration.

## Python inference

```python
from pathlib import Path
from moederncolbert_onnx import Encoder, maxsim

encoder = Encoder(Path('artifacts'), precision='q8')
query = encoder.encode(['Wie hoch ist die Zugspitze?'], 'query')[0]
documents = encoder.encode(['Die Zugspitze ist 2962 Meter hoch.'], 'document')
print(maxsim(query, documents[0]))
print(encoder.rerank('Wie hoch ist die Zugspitze?', ['Die Zugspitze ist 2962 Meter hoch.']))
```

## Architecture and source provenance

Pinned revision: `9d360b8b88ae11bea5784bf9012ff509fdd5c2da`.
The checkpoint contains 1,468,064 backbone parameters plus 16,384 projection
parameters: **1,484,448 total**. The exported graph includes the projection head
and per-token L2 normalization. ONNX opset 17, inputs `input_ids` and
`attention_mask` as INT64 `[batch, sequence]`, output `token_embeddings` as
FP32 `[batch, sequence, 128]`.

The custom model is exported through a tensor-only implementation of its two
attention/MoE layers. All eight routed experts are evaluated, then the learned
top-2 gates select their contribution; the shared expert is preserved. This
avoids Triton, Python data-dependent export paths and custom runtime operators,
but does not preserve sparse expert compute savings. The portable graph caps
positions at 512. This project makes no 32k-context claim.

The upstream reference uses Sentence Transformers' `MultiVectorEncoder` with
local custom code from the pinned, checksum-verified checkpoint. Build-time
parity compares both its full forward path and public query/document encoding.
The effective tokenizer is saved after the reference promotes the `[Q] ` and
`[D] ` markers to special tokens. JavaScript inference needs neither PyTorch nor
Python nor execution of remote model code.

Models and upstream/adapted model code are Apache-2.0; original project code is
MIT. See `NOTICE`, `LICENSE`, and `artifacts/MODEL-LICENSE.txt`.

Primary references: [pinned model files](https://huggingface.co/VAGOsolutions/SauerkrautLM-MoEdernColBERT-1.5M/tree/9d360b8b88ae11bea5784bf9012ff509fdd5c2da),
[ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html),
and [defuss-vectorsearch](https://github.com/kyr0/defuss-vectorsearch).
