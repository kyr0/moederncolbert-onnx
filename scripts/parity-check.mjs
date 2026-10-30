import { maxSim } from '../dist/index.mjs';

function check(condition, message) { if (!condition) throw new Error(message); }
function equal(actual, expected, label) {
  check(JSON.stringify(actual) === JSON.stringify(expected), `Mismatch: ${label}`);
}
export async function verifyEncoders(factory, fixture) {
  const reports = {};
  for (const precision of ['fp32', 'q8']) {
    const encoder = await factory(precision);
    let maxError = 0;
    let numbersCompared = 0;
    const caseErrors = [];
    try {
      equal(encoder.manifest.models[precision].sha256, fixture.graph_hashes[precision], 'fixture graph hash');
      for (const [caseIndex, test] of fixture.cases.entries()) {
        let caseError = 0;
        const options = { truncate: test.truncate, batchSize: test.texts.length };
        const prep = encoder.preprocessor.prepare(test.texts, test.kind, options);
        for (const key of ['inputIds', 'attentionMask', 'scoreMask', 'truncated'])
          equal(prep[key], test[key], `${precision} case ${caseIndex} ${key}`);
        const encoded = await encoder.encode(test.texts, test.kind, options);
        for (let row = 0; row < encoded.length; row++) {
          const expected = test.outputs[precision][row].flat();
          check(encoded[row].data.length === expected.length, 'Embedding length mismatch');
          for (let i = 0; i < expected.length; i++) {
            const difference = Math.abs(encoded[row].data[i] - expected[i]);
            check(Number.isFinite(difference), 'Non-finite output');
            maxError = Math.max(maxError, difference);
            caseError = Math.max(caseError, difference);
          }
          numbersCompared += expected.length;
        }
        caseErrors.push({ caseIndex, kind: test.kind, maxError: caseError });
      }
      const queries = await encoder.encodeQueries(fixture.corpus.queries.map(q => q.text));
      const docs = await encoder.encodeDocuments(fixture.corpus.documents);
      let maxScoreError = 0;
      let top1Matches = 0;
      for (let q = 0; q < queries.length; q++) {
        const scores = docs.map(doc => maxSim(queries[q], doc));
        const expected = fixture.scores[precision][q];
        for (let d = 0; d < docs.length; d++) maxScoreError = Math.max(maxScoreError, Math.abs(scores[d] - expected[d]));
        if (scores.indexOf(Math.max(...scores)) === expected.indexOf(Math.max(...expected))) top1Matches++;
      }
      check(maxError < 1e-4 && maxScoreError < 1e-3, `${precision} cross-runtime parity failed: ${JSON.stringify({ maxError, maxScoreError, top1Matches, caseErrors })}`);
      check(top1Matches === queries.length, `${precision} top-1 differs from Python`);
      reports[precision] = { passed: true, cases: fixture.cases.length, numbersCompared,
        maxAbsoluteError: maxError, maxScoreError, top1Matches, queries: queries.length,
        graphSha256: encoder.manifest.models[precision].sha256 };
    } finally { await encoder.dispose(); }
  }
  return reports;
}
