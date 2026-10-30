// Interactive demo. One encoder per model and precision stays alive, and document vectors are cached per
// encoder. VERIFIED: loading the 27M FP32 graph took 4.7 s in Chromium WASM (make e2e status line).
import { createBrowserEncoder } from '../src/browser';
import { rankEncoded, type ColBERTEncoder, type Precision, type TokenMatrix } from '../src/index';
import './style.css';

interface ModelEntry { repo: string; modelId: string; precisions: Precision[] }
const LABELS: Record<Precision, string> = {
  fp32: 'FP32 · reference export', fp16: 'FP16', q8: 'Q8 · INT8 weights, FP32 math',
  int8: 'INT8 · dynamic activations', q4: 'Q4 · 4-bit weights',
};
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const status = $('status');
const button = $<HTMLButtonElement>('run');
const passages = $<HTMLTextAreaElement>('documents');
const modelSelect = $<HTMLSelectElement>('model');
const precisionSelect = $<HTMLSelectElement>('precision');
const encoders = new Map<string, ColBERTEncoder>();
const cache = new Map<string, { key: string; vectors: TokenMatrix[] }>();
let models: ModelEntry[] = [];

function option(value: string, label: string): HTMLOptionElement {
  const element = document.createElement('option');
  element.value = value; element.textContent = label;
  return element;
}
function showPrecisions(): void {
  const model = models.find(m => m.repo === modelSelect.value)!;
  const previous = precisionSelect.value;
  precisionSelect.replaceChildren(...model.precisions.map(p => option(p, LABELS[p])));
  precisionSelect.value = model.precisions.includes(previous as Precision) ? previous : 'q8';
}
try {
  const [index, corpus] = await Promise.all(['/models/index.json', '/fixtures/retrieval.json'].map(async url => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    return response.json();
  }));
  models = index;
  modelSelect.replaceChildren(...models.map(m => option(m.repo, m.modelId.split('/')[1]!)));
  showPrecisions();
  passages.value = corpus.documents.join('\n');
  button.disabled = false;
  status.textContent = 'Ready. Models load on first use; all processing stays in this browser.';
} catch (error) { status.textContent = `Could not load models or samples: ${String(error)}`; }
modelSelect.addEventListener('change', showPrecisions);

button.addEventListener('click', async () => {
  button.disabled = true;
  const repo = modelSelect.value;
  const precision = precisionSelect.value as Precision;
  const id = `${repo}:${precision}`;
  const query = $<HTMLInputElement>('query').value;
  const documents = passages.value.split('\n').map(s => s.trim()).filter(Boolean);
  const truncate = $<HTMLInputElement>('truncate').checked;
  const results = $('results');
  results.replaceChildren();
  try {
    if (!documents.length) throw new Error('Add at least one candidate passage.');
    const loadStart = performance.now();
    let encoder = encoders.get(id);
    if (!encoder) {
      status.textContent = `Loading ${repo} ${precision.toUpperCase()} and the local WASM runtime…`;
      encoder = await createBrowserEncoder(`/models/${repo}/`, { precision, wasmPath: '/ort/' });
      encoders.set(id, encoder);
    }
    const loadMs = performance.now() - loadStart;
    status.textContent = 'Encoding and scoring…';
    const start = performance.now();
    const key = JSON.stringify([documents, truncate]);
    const cached = cache.get(id);
    const vectors = cached?.key === key ? cached.vectors : await encoder.encodeDocuments(documents, { truncate });
    cache.set(id, { key, vectors });
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
    status.textContent = `${repo} · ${precision.toUpperCase()} · ${documents.length} passages · ${(performance.now() - start).toFixed(0)} ms encode + score · ${loadMs.toFixed(0)} ms load · ${cached?.key === key ? 'cached documents' : 'fresh documents'}${truncatedCount ? ` · ${truncatedCount} inputs truncated` : ''}`;
  } catch (error) { status.textContent = `Error: ${error instanceof Error ? error.message : String(error)}`; }
  finally { button.disabled = false; }
});
