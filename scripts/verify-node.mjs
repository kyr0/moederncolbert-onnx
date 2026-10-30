import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { platform, arch } from 'node:os';
import { createNodeEncoder } from '../dist/node.mjs';
import { verifyEncoders } from './parity-check.mjs';

const fixture = JSON.parse(await readFile('artifacts/parity-fixture.json', 'utf8'));
const results = await verifyEncoders(precision => createNodeEncoder('artifacts', { precision }), fixture);
const report = { passed: true, runtime: process.version, platform: platform(), arch: arch(), results };
await mkdir('reports', { recursive: true });
await writeFile('reports/node-parity.json', JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
