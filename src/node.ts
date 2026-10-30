// Node.js entry: onnxruntime-node on the CPU. It reads files from disk; integrity checks stay in the core.
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import * as ort from 'onnxruntime-node';
import { createEncoder, type Precision } from './index.js';

export async function createNodeEncoder(directory: string, options: { precision?: Precision; threads?: number } = {}) {
  const threads = options.threads ?? 2;
  if (!Number.isInteger(threads) || threads < 1) throw new Error('threads must be a positive integer');
  return createEncoder(name => readFile(resolve(directory, name)), async model => {
    const session = await ort.InferenceSession.create(model, {
      executionProviders: ['cpu'], intraOpNumThreads: threads, interOpNumThreads: 1,
    });
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
  }, options.precision, `onnxruntime-node@${ort.env.versions.node}/cpu`);
}
