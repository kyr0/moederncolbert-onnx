// Cross-runtime check of every built upload folder in Node.js (onnxruntime-node, CPU).
import { readFile, readdir, mkdir, writeFile } from 'node:fs/promises';
import { platform, arch } from 'node:os';
import { createNodeEncoder } from '../dist/node.mjs';
import { verifyEncoders } from './parity-check.mjs';

const folders = (await readdir('output/hf', { withFileTypes: true })).filter(e => e.isDirectory()).map(e => e.name);
if (!folders.length) throw new Error('No upload folders in output/hf; run `make models` first');
for (const repo of folders.sort()) {
  const fixture = JSON.parse(await readFile(`output/parity/${repo}.json`, 'utf8'));
  const results = await verifyEncoders(precision => createNodeEncoder(`output/hf/${repo}`, { precision }), fixture);
  const report = { passed: true, repo, runtime: process.version, platform: platform(), arch: arch(), results };
  await mkdir(`reports/${fixture.model.slug}`, { recursive: true });
  await writeFile(`reports/${fixture.model.slug}/node-parity.json`, JSON.stringify(report, null, 2) + '\n');
  console.log(`${repo}: ${Object.entries(results).map(([p, r]) => `${p} ${r.parity} ok`).join(', ')}`);
}
