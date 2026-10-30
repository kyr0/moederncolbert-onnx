# MoEdernColBERT ONNX

This project converts two multilingual ColBERT retrievers,
[SauerkrautLM-MoEdernColBERT-1.5M](https://huggingface.co/VAGOsolutions/SauerkrautLM-MoEdernColBERT-1.5M)
and [SauerkrautLM-MoEdernColBERT-27M](https://huggingface.co/VAGOsolutions/SauerkrautLM-MoEdernColBERT-27M),
into ONNX graphs that run on CPU in Python, Node.js and browsers. It builds one Hugging Face upload
folder per model, checks every graph against the upstream PyTorch model and across runtimes, and
ships a small JavaScript and Python library for token encoding and MaxSim reranking.

## Upload folders

`make models` writes one folder per model into `output/hf/`, ready for `hf upload`:

| Folder | Hugging Face repo | Precisions |
|---|---|---|
| `output/hf/SauerkrautLM-MoEdernColBERT-1.5M-ONNX` | `kyr0/SauerkrautLM-MoEdernColBERT-1.5M-ONNX` | fp32, fp16, q8, int8 |
| `output/hf/SauerkrautLM-MoEdernColBERT-27M-ONNX` | `kyr0/SauerkrautLM-MoEdernColBERT-27M-ONNX` | fp32, fp16, q8, int8, q4 |

Each folder holds the graphs `model_<precision>.onnx`, the tokenizer, `runtime.json` (the
tokenization contract), `manifest.json` (size and SHA-256 of every file), `parity-report.json`,
the upstream Apache-2.0 `LICENSE`, and a generated model card `README.md` whose numbers come from
the parity report.

| Precision | Weights and math | Cross-runtime contract |
|---|---|---|
| fp32 | FP32 reference export | exact |
| fp16 | FP16 weights and math, FP32 inputs and outputs | quality |
| q8 | INT8 weights, FP32 math | exact |
| int8 | INT8 weights, dynamic INT8 activations | quality |
| q4 | 4-bit weights (MatMulNBits), FP32 math | exact |

An **exact** graph gives the same token vectors in Python, Node.js and browser ONNX Runtime, within
1e-4 per value. A **quality** graph depends on runtime kernels, so each runtime is checked against
its own FP32 output instead, and cached vectors carry the runtime in their fingerprint.
[ARCH.md](ARCH.md) explains both contracts.

The 1.5M folder ships no q4 graph. VERIFIED: its mean token cosine against FP32 measured 0.954 to
0.961 for block sizes 16 and 32, below the 0.98 gate that every quantized variant must pass.

Upload a folder after `make models` (or `make e2e`) has passed:

```sh
hf upload kyr0/SauerkrautLM-MoEdernColBERT-1.5M-ONNX output/hf/SauerkrautLM-MoEdernColBERT-1.5M-ONNX .
hf upload kyr0/SauerkrautLM-MoEdernColBERT-27M-ONNX output/hf/SauerkrautLM-MoEdernColBERT-27M-ONNX .
```

Graph hashes differ by build platform (see [ARCH.md](ARCH.md#reproducibility)), so upload each
folder exactly as one build produced it.

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), [Bun](https://bun.sh) and Node.js 22.12 or later.
On macOS the locked ONNX Runtime wheel needs macOS 14 on Apple Silicon. Export runs on CPU.

```sh
make setup        # locked deps, Chromium, JS library, both upload folders
make demo         # interactive reranking demo (Vite dev server)
make demo-node    # rerank four passages in Node.js with the 1.5M q8 graph
```

`make setup` downloads the pinned checkpoints (about 115 MB for both) into `upstream/<model>/` and
rejects any file whose SHA-256 differs from the hash pinned in `source.py`.

## Commands

| Command | What it does |
|---|---|
| `make models` | Build both upload folders: download, export, variants, parity gates, model card |
| `make test` | Python suite (builds both models in a temp dir), JS suite, Node.js cross-runtime check |
| `make lint` | ruff, oxlint, TypeScript |
| `make coverage` | Python coverage (gate floor 60 %) |
| `make e2e` | Build the wheel and the npm tarball, consume both like a user, drive the demo in Chromium |
| `make verify` | lint, test, coverage and e2e, as in CI |
| `make start` / `stop` / `status` / `log` | Serve the built demo on `http://127.0.0.1:4791` (`PORT=` overrides) |
| `make bench` | Median and p95 timings per precision on this machine |
| `make metrics` | Byte size of every built graph |

`MODELS=27m make models` limits a command to one model. The Python CLI does the same per step:

```sh
uv run --frozen moedern-onnx download --model 27m
uv run --frozen moedern-onnx export --model 27m
uv run --frozen moedern-onnx quantize --model 27m
uv run --frozen moedern-onnx parity --model 27m
uv run --frozen moedern-onnx demo --model 27m --precision q4 --query 'Wie hoch ist die Zugspitze?'
uv run --frozen moedern-onnx reference --model 27m   # small score fixture for consumers
uv run --frozen moedern-onnx bench --model 27m
```

`build` runs download, export, quantize and parity in one go. `--source`, `--artifacts`,
`--reports`, `--parity-fixture` and `--out` override the default paths.

## Results

VERIFIED on macOS 26.7.1 (Apple Silicon), Python 3.11.17, PyTorch 2.6.0, ONNX Runtime 1.31.1
(Python) and onnxruntime-node 1.30.0 on Node.js 26.11.1. Full numbers are in `reports/<model>/`.

The FP32 graphs match the upstream Sentence Transformers model within 1.0e-6 (1.5M) and 8.6e-7
(27M) per value, across tokenizer edge cases and batches of up to three 512-token sequences. For
the 27M model, those batches also reach upstream's grouped mixture-of-experts code path.

The other precisions are compared with FP32 on a fixture of 24 hand-written queries over 24
passages in seven languages:

| Model | Precision | Bytes | Mean token cosine | Same top-1 as FP32 | Top-5 overlap |
|---|---|---:|---:|---:|---:|
| 1.5M | fp32 | 6,294,481 | | | |
| 1.5M | fp16 | 3,239,914 | 0.9999 | 24/24 | 1.000 |
| 1.5M | q8 | 2,016,962 | 0.9971 | 24/24 | 0.975 |
| 1.5M | int8 | 2,113,134 | 0.9943 | 24/24 | 0.967 |
| 27M | fp32 | 108,852,703 | | | |
| 27M | fp16 | 54,519,032 | 0.9999 | 24/24 | 1.000 |
| 27M | q8 | 27,816,805 | 0.9991 | 24/24 | 0.992 |
| 27M | int8 | 27,944,234 | 0.9979 | 24/24 | 0.967 |
| 27M | q4 | 18,395,356 | 0.9852 | 24/24 | 0.908 |

Every precision of both models ranks the labeled passage first for all 24 queries. The fixture is
a regression check, not a retrieval benchmark, and says nothing about quality against other models.

In Node.js, every exact graph reproduces the Python token vectors with zero difference and every
MaxSim score within 2.1e-6. The fp16 and int8 graphs pass the same quality gate against Node.js's
own FP32 output.

In Chromium (headless Chrome for Testing 153, ONNX Runtime Web 1.30.0, single-thread WASM), every
exact graph stays within 1.1e-6 per value and 8.2e-6 per score of Python. Against the browser's own
FP32 output, fp16 reaches a mean token cosine of 0.9999 and int8 0.9913 (1.5M) and 0.9919 (27M).
The demo ranks the Zugspitze passage first for all nine model and precision pairs. Results are in
`reports/<model>/browser-parity.json`.

## JavaScript

```js
import { createNodeEncoder } from 'moederncolbert-onnx/node';
import { rankEncoded } from 'moederncolbert-onnx';

const encoder = await createNodeEncoder('output/hf/SauerkrautLM-MoEdernColBERT-27M-ONNX', { precision: 'q8' });
try {
  const passages = ['Die Zugspitze ist 2962 Meter hoch.', 'Paris is the capital of France.'];
  const documents = await encoder.encodeDocuments(passages); // cache these, keyed by encoder.fingerprint
  const [query] = await encoder.encodeQueries(['Wie hoch ist die Zugspitze?']);
  console.log(rankEncoded(query, documents));
} finally {
  await encoder.dispose();
}
```

The browser entry `createBrowserEncoder(folderUrl, { precision, wasmPath })` from
`moederncolbert-onnx/browser` runs the same code on ONNX Runtime Web (single-thread WASM). Both
check every file's SHA-256 against `manifest.json` before creating a session.

- Queries always use 32 positions; documents use up to 512.
- Longer inputs throw unless `truncate: true` is set, and each result reports whether it was truncated.
- `maxSim` rejects token matrices with different fingerprints, so vectors from different graphs,
  or from different runtimes for a quality graph, never mix.

## Python

```python
from pathlib import Path
from moederncolbert_onnx import Encoder

encoder = Encoder(Path("output/hf/SauerkrautLM-MoEdernColBERT-27M-ONNX"), "q8")
print(encoder.rerank("Wie hoch ist die Zugspitze?", ["Die Zugspitze ist 2962 Meter hoch.", "Paris is in France."]))
```

Each model card also contains a dependency-light Python example (onnxruntime, tokenizers, numpy)
that the test suite executes.

## License

Project code is MIT ([LICENSE](LICENSE)). The checkpoints, the converted graphs and the adapted
model code in `src/moederncolbert_onnx/portable.py` are Apache-2.0; see [NOTICE](NOTICE).
