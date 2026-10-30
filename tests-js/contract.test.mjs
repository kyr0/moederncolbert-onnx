import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { maxSim, createEncoder, rankEncoded } from '../dist/index.mjs';
import { createNodeEncoder } from '../dist/node.mjs';

const matrix = data => ({ data: Float32Array.from(data), tokenCount: data.length / 2,
  dimensions: 2, fingerprint: 'test', truncated: false });

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
    readFile(`artifacts/${name}`), async () => { backendCalled = true; throw new Error('unexpected'); }), /checksum/);
  assert.equal(backendCalled, false);
});

for (const precision of ['fp32', 'q8']) test(`${precision}: inference, cache, validation and disposal`, async () => {
  const encoder = await createNodeEncoder('artifacts', { precision });
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
    assert.equal((await encoder.encodeQueries(['still works']))[0].tokenCount, 32);
  } finally { await encoder.dispose(); }
  await encoder.dispose();
  await assert.rejects(encoder.encodeQueries(['x']), /disposed/);
});
