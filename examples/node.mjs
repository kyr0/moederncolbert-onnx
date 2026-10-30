import { createNodeEncoder } from '../dist/node.mjs';
import { rankEncoded } from '../dist/index.mjs';

const precision = process.argv[2] ?? 'q8';
const query = process.argv[3] ?? 'Wie hoch ist die Zugspitze?';
const documents = [
  'Die Zugspitze ist mit 2962 Metern der höchste Berg Deutschlands.',
  'Paris is the capital of France.',
  'A bicycle has two wheels.',
  'Die Alpen sind ein Gebirge in Europa.',
];
const encoder = await createNodeEncoder('artifacts', { precision });
try {
  // Precompute and cache these per document, keyed by encoder.fingerprint.
  const cached = await encoder.encodeDocuments(documents);
  const [encodedQuery] = await encoder.encodeQueries([query]);
  console.log({ precision, query, fingerprint: encoder.fingerprint });
  console.table(rankEncoded(encodedQuery, cached).map(r => ({ ...r, text: documents[r.index] })));
} finally { await encoder.dispose(); }
