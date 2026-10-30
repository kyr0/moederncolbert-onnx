// Cross-runtime harness shared by Node.js (verify-node.mjs) and Chromium (validation.html). It reads every
// threshold from the Python-written fixture, so the JavaScript side never re-declares a contract.
import { maxSim } from '../dist/index.mjs';

function check(condition, message) { if (!condition) throw new Error(message); }
function equal(actual, expected, label) {
  check(JSON.stringify(actual) === JSON.stringify(expected), `Mismatch: ${label}`);
}
const ranking = scores => scores.map((score, index) => [score, index])
  .sort((a, b) => b[0] - a[0] || a[1] - b[1]).map(([, index]) => index);

/** Mean token cosine between two lists of token matrices with identical shapes. */
function meanTokenCosine(reference, actual) {
  let sum = 0, count = 0;
  reference.forEach((matrix, m) => {
    const width = matrix.dimensions;
    for (let t = 0; t < matrix.tokenCount; t++) {
      let dot = 0, a = 0, b = 0;
      for (let k = 0; k < width; k++) {
        const x = matrix.data[t * width + k], y = actual[m].data[t * width + k];
        dot += x * y; a += x * x; b += y * y;
      }
      sum += dot / Math.sqrt(a * b); count++;
    }
  });
  return sum / count;
}

/** Python's golden outputs for exact precisions; the runtime's own FP32 for quality precisions.
 * VERIFIED: fp16 and int8 vectors differ between runtimes (contracts.py), so only quality is comparable. */
export async function verifyEncoders(factory, fixture) {
  check(fixture.schema_version === 2 && fixture.precisions[0] === 'fp32', 'Unsupported parity fixture');
  const reports = {};
  let fp32;
  for (const precision of fixture.precisions) {
    const contract = fixture.contracts[precision];
    const encoder = await factory(precision);
    try {
      equal(encoder.manifest.models[precision].sha256, fixture.graph_hashes[precision], 'fixture graph hash');
      const queries = await encoder.encodeQueries(fixture.corpus.queries.map(q => q.text));
      const documents = await encoder.encodeDocuments(fixture.corpus.documents);
      const scores = queries.map(q => documents.map(d => maxSim(q, d)));
      const report = { passed: true, parity: contract.parity, graphSha256: fixture.graph_hashes[precision],
        fingerprint: encoder.fingerprint };
      if (contract.parity === 'exact') {
        const tolerance = fixture.exact_tolerance;
        let maxError = 0, numbersCompared = 0;
        for (const [caseIndex, test] of fixture.cases.entries()) {
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
            }
            numbersCompared += expected.length;
          }
        }
        let maxScoreError = 0, top1Matches = 0;
        scores.forEach((row, q) => {
          const expected = fixture.scores[precision][q];
          row.forEach((score, d) => { maxScoreError = Math.max(maxScoreError, Math.abs(score - expected[d])); });
          if (ranking(row)[0] === ranking(expected)[0]) top1Matches++;
        });
        check(maxError <= tolerance.token && maxScoreError <= tolerance.score,
          `${precision} cross-runtime parity failed: ${JSON.stringify({ maxError, maxScoreError })}`);
        check(top1Matches === queries.length, `${precision} top-1 differs from Python`);
        Object.assign(report, { cases: fixture.cases.length, numbersCompared, maxAbsoluteError: maxError,
          maxScoreError, top1Matches, queries: queries.length });
      } else {
        check(fp32, 'FP32 must be verified before quality precisions');
        const meanCosine = meanTokenCosine([...fp32.queries, ...fp32.documents], [...queries, ...documents]);
        let top1 = 0, top5 = 0;
        scores.forEach((row, q) => {
          const a = ranking(row), b = ranking(fp32.scores[q]);
          if (a[0] === b[0]) top1++;
          top5 += a.slice(0, 5).filter(index => b.slice(0, 5).includes(index)).length / 5;
        });
        const quality = { meanTokenCosine: meanCosine, top1Agreement: top1 / queries.length,
          top5Overlap: top5 / queries.length };
        check(quality.meanTokenCosine >= contract.mean_token_cosine && quality.top1Agreement >= contract.top1_agreement
          && quality.top5Overlap >= contract.top5_overlap,
          `${precision} quality gate failed against this runtime's FP32: ${JSON.stringify(quality)}`);
        Object.assign(report, quality);
      }
      if (precision === 'fp32') fp32 = { queries, documents, scores };
      reports[precision] = report;
    } finally { await encoder.dispose(); }
  }
  return reports;
}
