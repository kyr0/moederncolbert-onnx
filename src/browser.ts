// Browser entry: ONNX Runtime Web on single-thread WASM. VERIFIED: the parity fixture passes on this
// configuration in Chromium for every precision of both models (make e2e).
// Multi-threaded WASM needs cross-origin isolation headers that a static host may not send.
import * as ort from 'onnxruntime-web/wasm';
import { createEncoder, type Precision } from './index.js';

/** WASM CPU implementation. Configure wasmPath before creating any session. */
export async function createBrowserEncoder(baseUrl: string | URL,
  options: { precision?: Precision; wasmPath?: string } = {}) {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  if (options.wasmPath) ort.env.wasm.wasmPaths = options.wasmPath;
  const base = new URL(baseUrl, globalThis.location.href);
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return createEncoder(async name => {
    const response = await fetch(new URL(name, base));
    if (!response.ok) throw new Error(`Cannot load ${name}: HTTP ${response.status}`);
    return new Uint8Array(await response.arrayBuffer());
  }, async model => {
    const session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'] });
    return {
      async run(ids, mask, shape) {
        const outputs = await session.run({
          input_ids: new ort.Tensor('int64', ids, shape),
          attention_mask: new ort.Tensor('int64', mask, shape),
        });
        return outputs.token_embeddings!.data as Float32Array;
      },
      dispose: () => session.release(),
    };
  }, options.precision, `onnxruntime-web@${ort.env.versions.web}/wasm`);
}
