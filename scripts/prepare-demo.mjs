import { mkdir, copyFile, readFile } from 'node:fs/promises';

await mkdir('public/models', { recursive: true });
await mkdir('public/ort', { recursive: true });
await mkdir('public/fixtures', { recursive: true });
const manifest = JSON.parse(await readFile('artifacts/manifest.json', 'utf8'));
for (const name of ['manifest.json', 'MODEL-LICENSE.txt', ...Object.keys(manifest.files)])
  await copyFile(`artifacts/${name}`, `public/models/${name}`);
for (const suffix of ['mjs', 'wasm']) await copyFile(
  `node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.${suffix}`,
  `public/ort/ort-wasm-simd-threaded.${suffix}`);
await copyFile('third-party/ONNXRUNTIME-LICENSE.txt', 'public/ort/LICENSE.txt');
await copyFile('artifacts/parity-fixture.json', 'public/fixtures/parity.json');
await copyFile('fixtures/retrieval.json', 'public/fixtures/retrieval.json');
console.log('Prepared local model, tokenizer, fixture, and WASM assets. No CDN needed.');
