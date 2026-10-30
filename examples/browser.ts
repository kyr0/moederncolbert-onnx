import { createBrowserEncoder } from '../src/browser';
import { rankEncoded, type ColBERTEncoder, type Precision, type TokenMatrix } from '../src/index';
import './style.css';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const status = $('status');
const button = $<HTMLButtonElement>('run');
const passages = $<HTMLTextAreaElement>('documents');
const encoders = new Map<Precision, ColBERTEncoder>();
const cache = new Map<Precision, { key: string; vectors: TokenMatrix[] }>();
try {
  const response = await fetch('/fixtures/retrieval.json');
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const corpus = await response.json();
  passages.value = corpus.documents.join('\n');
  status.textContent = 'Ready. Models load on first use; all processing stays in this browser.';
} catch (error) { status.textContent = `Could not load samples: ${String(error)}`; }

button.addEventListener('click', async () => {
  button.disabled = true;
  const precision = $<HTMLSelectElement>('precision').value as Precision;
  const query = $<HTMLInputElement>('query').value;
  const documents = passages.value.split('\n').map(s => s.trim()).filter(Boolean);
  const truncate = $<HTMLInputElement>('truncate').checked;
  const results = $('results');
  results.replaceChildren();
  try {
    if (!documents.length) throw new Error('Add at least one candidate passage.');
    const loadStart = performance.now();
    let encoder = encoders.get(precision);
    if (!encoder) {
      status.textContent = `Loading ${precision.toUpperCase()} model and local WASM runtime…`;
      encoder = await createBrowserEncoder('/models/', { precision, wasmPath: '/ort/' });
      encoders.set(precision, encoder);
    }
    const loadMs = performance.now() - loadStart;
    status.textContent = 'Encoding and scoring…';
    const start = performance.now();
    const key = JSON.stringify([documents, truncate]);
    const cached = cache.get(precision);
    const vectors = cached?.key === key ? cached.vectors : await encoder.encodeDocuments(documents, { truncate });
    cache.set(precision, { key, vectors });
    const [encodedQuery] = await encoder.encodeQueries([query], { truncate });
    for (const hit of rankEncoded(encodedQuery!, vectors)) {
      const row = document.createElement('li');
      const score = document.createElement('strong');
      score.textContent = hit.score.toFixed(3);
      const text = document.createElement('span');
      text.textContent = documents[hit.index]!;
      row.append(score, text); results.append(row);
    }
    const truncatedCount = vectors.filter(v => v.truncated).length + Number(encodedQuery!.truncated);
    status.textContent = `${precision.toUpperCase()} · ${documents.length} passages · ${(performance.now() - start).toFixed(0)} ms encode + score · ${loadMs.toFixed(0)} ms load · ${cached?.key === key ? 'cached documents' : 'fresh documents'}${truncatedCount ? ` · ${truncatedCount} inputs truncated` : ''}`;
  } catch (error) { status.textContent = `Error: ${error instanceof Error ? error.message : String(error)}`; }
  finally { button.disabled = false; }
});
