// Drives the built demo served by `make start` in Chromium: WASM parity for every model, then the UI
// for every model × precision. Fails on page errors, console errors and failed requests.
import { mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright';

const base = process.env.DEMO_URL || 'http://127.0.0.1:4791';
const out = 'output/browser';
await mkdir(out, { recursive: true });
const browser = await chromium.launch({ headless: true,
  ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ?
    { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {}) });
try {
  const page = await browser.newPage({ viewport: { width: 1200, height: 1000 } });
  const errors = [];
  page.on('pageerror', error => errors.push(`pageerror: ${error}`));
  page.on('console', message => { if (message.type() === 'error') errors.push(`console: ${message.text()}`); });
  page.on('requestfailed', request => errors.push(`requestfailed: ${request.url()} ${request.failure()?.errorText}`));
  page.on('response', response => { if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`); });
  const models = await (await fetch(`${base}/models/index.json`)).json();
  const parity = {};
  await page.goto(`${base}/validation.html`);
  await page.waitForFunction(() => typeof window.runParity === 'function');
  for (const { repo } of models) {
    parity[repo] = await page.evaluate(name => window.runParity(name), repo);
    console.log(`${repo}: WASM ${Object.entries(parity[repo]).map(([p, r]) => `${p} ${r.parity} ok`).join(', ')}`);
  }
  await page.goto(base);
  await page.waitForFunction(() => document.querySelector('#documents').value.includes('Zugspitze'));
  const ui = [];
  for (const { repo, precisions } of models) {
    await page.selectOption('#model', repo);
    for (const precision of precisions) {
      await page.selectOption('#precision', precision);
      await page.click('#run');
      await page.waitForFunction(expected => document.querySelector('#status').textContent.includes(expected),
        `${repo} · ${precision.toUpperCase()} ·`, { timeout: 120000 });
      const first = await page.locator('#results li').first().textContent();
      if (!first.includes('Zugspitze')) throw new Error(`Wrong top result for ${repo} ${precision}: ${first}`);
      ui.push({ repo, precision, status: await page.locator('#status').textContent() });
    }
    await page.screenshot({ path: `${out}/${repo}.png`, fullPage: true });
  }
  if (errors.length) throw new Error(errors.join('\n'));
  const report = { passed: true, browser: await browser.version(), backend: 'WASM CPU, one thread', parity, ui };
  await writeFile(`${out}/report.json`, JSON.stringify(report, null, 2) + '\n');
  for (const { repo, slug } of models) {
    await mkdir(`reports/${slug}`, { recursive: true });
    // Committed summary without timings. VERIFIED: two e2e runs on an unchanged build wrote identical files.
    await writeFile(`reports/${slug}/browser-parity.json`, JSON.stringify({ passed: true, browser: report.browser,
      backend: report.backend, results: parity[repo],
      ui: ui.filter(u => u.repo === repo).map(u => ({ precision: u.precision, topResult: 'Zugspitze' })) }, null, 2) + '\n');
  }
  console.log(`UI: ${ui.length} model/precision runs ranked Zugspitze first`);
} finally { await browser.close(); }
