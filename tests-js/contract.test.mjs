import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { maxSim, createEncoder, rankEncoded, PRECISIONS } from '../dist/index.mjs';
import { createNodeEncoder } from '../dist/node.mjs';

const ROOT = 'output/hf';
const folders = (await readdir(ROOT, { withFileTypes: true })).filter(e => e.isDirectory())
  .map(e => `${ROOT}/${e.name}`).sort();
const manifests = Object.fromEntries(await Promise.all(folders.map(async folder =>
  [folder, JSON.parse(await readFile(`${folder}/manifest.json`, 'utf8'))])));
const small = folders.find(folder => folder.includes('-1.5M-'));

const matrix = data => ({ data: Float32Array.from(data), tokenCount: data.length / 2,
  dimensions: 2, fingerprint: 'test', truncated: false });

test('both upload folders are built', () => {
  assert.deepEqual(folders.map(f => f.split('/').pop()),
    ['SauerkrautLM-MoEdernColBERT-1.5M-ONNX', 'SauerkrautLM-MoEdernColBERT-27M-ONNX']);
});

test('MaxSim handles negative similarities and stable ties', () => {
  const query = matrix([1, 0, 0, 1]);
  const document = matrix([-1, -2, -3, -1]);
  assert.equal(maxSim(query, document), -2);
  assert.deepEqual(rankEncoded(query, [document, document]).map(x => x.index), [0, 1]);
  assert.throws(() => maxSim(query, { ...document, fingerprint: 'other' }), /fingerprint/);
  assert.throws(() => maxSim(query, matrix([])), /matrix/);
  assert.throws(() => maxSim(query, matrix([NaN, 0])), /matrix/);
});

test('integrity verification precedes backend creation', async () => {
  let backendCalled = false;
  await assert.rejects(createEncoder(async name => name.endsWith('.onnx') ? new Uint8Array(3) :
    readFile(`${small}/${name}`), async () => { backendCalled = true; throw new Error('unexpected'); }), /checksum/);
  assert.equal(backendCalled, false);
});

test('unknown and unshipped precisions fail before loading a graph', async () => {
  const load = name => readFile(`${small}/${name}`);
  await assert.rejects(createEncoder(load, async () => assert.fail('no backend'), 'bf16'), /precision must be one of/);
  // The 1.5M folder ships no q4 graph (it fails the quality gate).
  await assert.rejects(createEncoder(load, async () => assert.fail('no backend'), 'q4'), /no q4 graph; available/);
});

test('fingerprints name the runtime only for quality-contract precisions', async () => {
  for (const folder of folders) for (const [precision, model] of Object.entries(manifests[folder].models)) {
    const encoder = await createNodeEncoder(folder, { precision });
    try {
      const base = `${manifests[folder].revision}:${model.sha256}`;
      if (model.parity === 'exact') assert.equal(encoder.fingerprint, base);
      else assert.match(encoder.fingerprint, new RegExp(`^${base}:onnxruntime-node@\\d+\\.\\d+\\.\\d+/cpu$`));
    } finally { await encoder.dispose(); }
  }
  assert.deepEqual([...PRECISIONS], ['fp32', 'fp16', 'q8', 'int8', 'q4']);
});

for (const folder of folders) for (const precision of Object.keys(manifests[folder].models))
  test(`${folder.split('/').pop()} ${precision}: inference, validation and disposal`, async () => {
    const encoder = await createNodeEncoder(folder, { precision });
    try {
      const documents = ['Die Zugspitze ist 2962 Meter hoch.', 'Paris is in France.'];
      const [query, docs] = await Promise.all([
        encoder.encodeQueries(['Wie hoch ist die Zugspitze?']), encoder.encodeDocuments(documents),
      ]);
      assert.equal(query[0].tokenCount, 32);
      assert.equal(rankEncoded(query[0], docs)[0].index, 0);
      assert.equal((await encoder.rerank('Wie hoch ist die Zugspitze?', documents, { topK: 1 }))[0].index, 0);
      assert.deepEqual(await encoder.encodeDocuments([]), []);
      assert.deepEqual(await encoder.rerank('x', []), []);
      await assert.rejects(encoder.encodeQueries(['the '.repeat(100)]), /limit/);
      assert.equal((await encoder.encodeQueries(['the '.repeat(100)], { truncate: true }))[0].truncated, true);
      await assert.rejects(encoder.encodeDocuments(['x'], { batchSize: 0 }), /batchSize/);
      await assert.rejects(encoder.rerank('x', documents, { topK: -1 }), /topK/);
      await assert.rejects(encoder.encode(['x'], 'invalid'), /kind/);
    } finally { await encoder.dispose(); }
    await encoder.dispose();
    await assert.rejects(encoder.encodeQueries(['x']), /disposed/);
  });
