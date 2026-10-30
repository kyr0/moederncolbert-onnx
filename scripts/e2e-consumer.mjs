// Runs inside a clean project that installed the packed tarball: node consumer.mjs <repo root>.
import { readFile, readdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { createNodeEncoder } from 'moederncolbert-onnx/node';
import { rankEncoded } from 'moederncolbert-onnx';

const root = process.argv[2];
const input = JSON.parse(await readFile(join(root, 'input/rerank.json'), 'utf8'));
const folders = (await readdir(join(root, 'output/hf'), { withFileTypes: true })).filter(e => e.isDirectory());
if (!folders.length) throw new Error('No upload folders to consume');
for (const { name: repo } of folders) {
  const folder = join(root, 'output/hf', repo);
  const manifest = JSON.parse(await readFile(join(folder, 'manifest.json'), 'utf8'));
  const results = {};
  for (const precision of Object.keys(manifest.models)) {
    const encoder = await createNodeEncoder(folder, { precision });
    try {
      const documents = await encoder.encodeDocuments(input.passages);
      const queries = await encoder.encodeQueries(input.queries.map(q => q.text));
      results[precision] = queries.map((query, i) => {
        const ranked = rankEncoded(query, documents);
        if (ranked[0].index !== input.queries[i].expected)
          throw new Error(`${repo} ${precision}: "${input.queries[i].text}" ranked passage ${ranked[0].index} first`);
        return { query: input.queries[i].text, ranking: ranked };
      });
    } finally { await encoder.dispose(); }
  }
  await writeFile(join(root, `output/e2e-node-${repo}.json`), JSON.stringify({ repo, results }, null, 2) + '\n');
  console.log(`${repo}: ${Object.keys(results).join(', ')} ranked every input query correctly`);
}
