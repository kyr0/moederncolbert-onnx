// Copies every built upload folder, its parity fixture and the WASM runtime into public/. No CDN needed.
import { mkdir, copyFile, readFile, readdir, rm, writeFile } from 'node:fs/promises';

await rm('public', { recursive: true, force: true });
for (const dir of ['public/models', 'public/ort', 'public/fixtures']) await mkdir(dir, { recursive: true });
const models = [];
const folders = (await readdir('output/hf', { withFileTypes: true })).filter(e => e.isDirectory()).map(e => e.name);
if (!folders.length) throw new Error('No upload folders in output/hf; run `make models` first');
for (const repo of folders.sort()) {
  const manifest = JSON.parse(await readFile(`output/hf/${repo}/manifest.json`, 'utf8'));
  await mkdir(`public/models/${repo}`, { recursive: true });
  for (const name of ['manifest.json', ...Object.keys(manifest.files)])
    await copyFile(`output/hf/${repo}/${name}`, `public/models/${repo}/${name}`);
  await copyFile(`output/parity/${repo}.json`, `public/fixtures/${repo}.json`);
  const { model } = JSON.parse(await readFile(`output/parity/${repo}.json`, 'utf8'));
  models.push({ repo, slug: model.slug, modelId: manifest.model_id, precisions: Object.keys(manifest.models) });
}
await writeFile('public/models/index.json', JSON.stringify(models, null, 2) + '\n');
for (const suffix of ['mjs', 'wasm']) await copyFile(
  `node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.${suffix}`,
  `public/ort/ort-wasm-simd-threaded.${suffix}`);
await copyFile('third-party/ONNXRUNTIME-LICENSE.txt', 'public/ort/LICENSE.txt');
await copyFile('fixtures/retrieval.json', 'public/fixtures/retrieval.json');
console.log(`Prepared ${models.map(m => m.repo).join(', ')} with parity fixtures and local WASM assets.`);
