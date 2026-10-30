import { Tokenizer } from '@huggingface/tokenizers';

export type Precision = 'fp32' | 'q8';
export type Kind = 'query' | 'document';
export interface RuntimeConfig {
  dimensions: number; query_length: number; max_document_length: number;
  prefixes: Record<Kind, string>; pad_token_id: number; sep_token_id: number;
  skip_token_ids: number[];
}
export interface Manifest {
  schema_version: number; model_id: string; revision: string;
  files: Record<string, { bytes: number; sha256: string }>;
  models: Record<Precision, { file: string; bytes: number; sha256: string;
    quantization?: { mode: string } }>;
}
export interface TokenMatrix {
  data: Float32Array; tokenCount: number; dimensions: number;
  fingerprint: string; truncated: boolean;
}
export interface EncodeOptions { batchSize?: number; truncate?: boolean; documentLength?: number }
export interface PreparedBatch {
  inputIds: number[][]; attentionMask: number[][]; scoreMask: boolean[][]; truncated: boolean[];
}
export interface RankedDocument { index: number; score: number; text: string }
export type ArtifactLoader = (filename: string) => Promise<Uint8Array>;
export interface Backend {
  run(ids: BigInt64Array, mask: BigInt64Array, shape: [number, number]): Promise<Float32Array>;
  dispose(): Promise<void>;
}
export type BackendFactory = (model: Uint8Array) => Promise<Backend>;

export async function sha256(bytes: Uint8Array): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest('SHA-256', new Uint8Array(bytes).buffer);
  return [...new Uint8Array(digest)].map(x => x.toString(16).padStart(2, '0')).join('');
}

export class Preprocessor {
  private readonly tokenizer: Tokenizer;
  private readonly skip: Set<number>;
  constructor(tokenizerJson: object, tokenizerConfig: object, readonly config: RuntimeConfig) {
    this.tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
    this.skip = new Set(config.skip_token_ids);
  }
  prepare(texts: string[], kind: Kind, options: EncodeOptions = {}): PreparedBatch {
    if (kind !== 'query' && kind !== 'document') throw new Error('kind must be query or document');
    if (!Array.isArray(texts) || !texts.length || texts.some(x => typeof x !== 'string'))
      throw new Error('Expected a nonempty array of strings');
    const documentLength = options.documentLength ?? this.config.max_document_length;
    if (!Number.isInteger(documentLength) || documentLength < 3 ||
        documentLength > this.config.max_document_length) throw new Error('documentLength must be in [3, 512]');
    const limit = kind === 'query' ? this.config.query_length : documentLength;
    const truncated: boolean[] = [];
    const rows = texts.map(text => {
      let ids = this.tokenizer.encode(this.config.prefixes[kind] + text).ids;
      const over = ids.length > limit;
      truncated.push(over);
      if (over) {
        if (!options.truncate) throw new Error(`${kind} has ${ids.length} tokens; limit is ${limit}; set truncate: true explicitly`);
        ids = [...ids.slice(0, limit - 1), this.config.sep_token_id];
      }
      return ids;
    });
    const width = kind === 'query' ? limit : Math.max(...rows.map(row => row.length));
    const inputIds = rows.map(row => [...row, ...Array<number>(width - row.length).fill(this.config.pad_token_id)]);
    const attentionMask = rows.map(row => Array.from({ length: width }, (_, i) => Number(i < row.length)));
    const scoreMask = inputIds.map((row, b) => row.map((id, i) => kind === 'query' ||
      (attentionMask[b]![i] === 1 && !this.skip.has(id))));
    return { inputIds, attentionMask, scoreMask, truncated };
  }
}

function validateMatrix(matrix: TokenMatrix): void {
  if (!Number.isInteger(matrix.tokenCount) || matrix.tokenCount < 1 ||
      !Number.isInteger(matrix.dimensions) || matrix.dimensions < 1 ||
      matrix.data.length !== matrix.tokenCount * matrix.dimensions ||
      matrix.data.some(value => !Number.isFinite(value))) throw new Error('Invalid token matrix');
}

/** Sum of maximum token dot products. Padding must already have been removed. */
export function maxSim(query: TokenMatrix, document: TokenMatrix): number {
  validateMatrix(query); validateMatrix(document);
  if (query.dimensions !== document.dimensions) throw new Error('Token dimensions differ');
  if (query.fingerprint !== document.fingerprint) throw new Error('Cannot mix model/precision fingerprints');
  const width = query.dimensions;
  let score = 0;
  for (let q = 0; q < query.tokenCount; q++) {
    let best = -Infinity;
    for (let d = 0; d < document.tokenCount; d++) {
      let dot = 0;
      for (let k = 0; k < width; k++) dot += query.data[q * width + k]! * document.data[d * width + k]!;
      best = Math.max(best, dot);
    }
    score += best;
  }
  return score;
}

/** Cache these document matrices in a sidecar, separate from the dense ANN index. */
export function rankEncoded(query: TokenMatrix, documents: TokenMatrix[]): { index: number; score: number }[] {
  return documents.map((doc, index) => ({ index, score: maxSim(query, doc) }))
    .sort((a, b) => b.score - a.score || a.index - b.index);
}

export class ColBERTEncoder {
  private closed = false;
  private queue: Promise<unknown> = Promise.resolve();
  readonly fingerprint: string;
  constructor(readonly preprocessor: Preprocessor, readonly manifest: Manifest,
    readonly precision: Precision, private readonly backend: Backend) {
    this.fingerprint = `${manifest.revision}:${manifest.models[precision].sha256}`;
  }
  private serial<T>(operation: () => Promise<T>): Promise<T> {
    if (this.closed) return Promise.reject(new Error('Encoder is disposed'));
    const next = this.queue.then(operation);
    this.queue = next.catch(() => undefined);
    return next;
  }
  encode(texts: string[], kind: Kind, options: EncodeOptions = {}): Promise<TokenMatrix[]> {
    return this.serial(async () => {
      if (!Array.isArray(texts) || texts.some(x => typeof x !== 'string')) throw new Error('texts must be an array of strings');
      if (kind !== 'query' && kind !== 'document') throw new Error('kind must be query or document');
      const size = options.batchSize ?? 4;
      if (!Number.isInteger(size) || size < 1) throw new Error('batchSize must be a positive integer');
      const results: TokenMatrix[] = [];
      const dimensions = this.preprocessor.config.dimensions;
      for (let start = 0; start < texts.length; start += size) {
        const batch = this.preprocessor.prepare(texts.slice(start, start + size), kind, options);
        const width = batch.inputIds[0]!.length;
        const output = await this.backend.run(BigInt64Array.from(batch.inputIds.flat(), BigInt),
          BigInt64Array.from(batch.attentionMask.flat(), BigInt), [batch.inputIds.length, width]);
        if (output.length !== batch.inputIds.length * width * dimensions ||
            output.some(x => !Number.isFinite(x))) throw new Error('Invalid encoder output');
        for (let b = 0; b < batch.inputIds.length; b++) {
          const keep = batch.scoreMask[b]!;
          const tokenCount = keep.filter(Boolean).length;
          const data = new Float32Array(tokenCount * dimensions);
          let target = 0;
          for (let t = 0; t < width; t++) if (keep[t]) {
            const source = (b * width + t) * dimensions;
            data.set(output.subarray(source, source + dimensions), target);
            target += dimensions;
          }
          results.push({ data, dimensions, tokenCount, fingerprint: this.fingerprint,
            truncated: batch.truncated[b]! });
        }
      }
      return results;
    });
  }
  encodeQueries(texts: string[], options?: EncodeOptions): Promise<TokenMatrix[]> {
    return this.encode(texts, 'query', options);
  }
  encodeDocuments(texts: string[], options?: EncodeOptions): Promise<TokenMatrix[]> {
    return this.encode(texts, 'document', options);
  }
  async rerank(query: string, documents: string[], options: EncodeOptions & { topK?: number } = {}): Promise<RankedDocument[]> {
    if (options.topK !== undefined && (!Number.isInteger(options.topK) || options.topK < 0))
      throw new Error('topK must be a nonnegative integer');
    if (this.closed) throw new Error('Encoder is disposed');
    if (!documents.length) return [];
    const q = (await this.encodeQueries([query], options))[0]!;
    const docs = await this.encodeDocuments(documents, options);
    return rankEncoded(q, docs).slice(0, options.topK ?? docs.length)
      .map(result => ({ ...result, text: documents[result.index]! }));
  }
  async dispose(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    await this.queue;
    await this.backend.dispose();
  }
}

export async function createEncoder(load: ArtifactLoader, createBackend: BackendFactory,
  precision: Precision = 'q8'): Promise<ColBERTEncoder> {
  if (precision !== 'fp32' && precision !== 'q8') throw new Error('precision must be fp32 or q8');
  const decode = (bytes: Uint8Array) => JSON.parse(new TextDecoder().decode(bytes));
  const manifest = decode(await load('manifest.json')) as Manifest;
  if (manifest.schema_version !== 1 || !manifest.models[precision]) throw new Error('Unsupported manifest');
  const graph = manifest.models[precision].file;
  const names = [graph, 'runtime.json', 'tokenizer.json', 'tokenizer_config.json'];
  const files = await Promise.all(names.map(async name => {
    if (!/^[a-zA-Z0-9_.-]+$/.test(name)) throw new Error('Unsafe artifact filename');
    const bytes = await load(name);
    const expected = manifest.files[name];
    if (!expected || bytes.length !== expected.bytes || await sha256(bytes) !== expected.sha256)
      throw new Error(`Artifact checksum mismatch: ${name}`);
    return bytes;
  }));
  const preprocessor = new Preprocessor(decode(files[2]!), decode(files[3]!), decode(files[1]!));
  return new ColBERTEncoder(preprocessor, manifest, precision, await createBackend(files[0]!));
}
