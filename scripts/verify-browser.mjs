import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright';

const server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', 'preview', '--host', '127.0.0.1', '--port', '4173', '--strictPort'], { stdio: ['ignore', 'pipe', 'pipe'] });
let serverOutput = '';
server.stdout.on('data', data => { serverOutput += data; });
server.stderr.on('data', data => { serverOutput += data; });
let browser;
try {
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Server startup timed out: ${serverOutput}`)), 15000);
    server.on('exit', code => { clearTimeout(timer); reject(new Error(`Server exited ${code}: ${serverOutput}`)); });
    server.stdout.on('data', () => { if (serverOutput.includes('127.0.0.1:4173')) { clearTimeout(timer); resolve(); } });
  });
  browser = await chromium.launch({ headless: true,
    ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ?
      { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {}) });
  const page = await browser.newPage({ viewport: { width: 1200, height: 1000 } });
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  page.on('response', response => { if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`); });
  await page.goto('http://127.0.0.1:4173/validation.html');
  await page.waitForFunction(() => typeof window.runParity === 'function');
  const results = await page.evaluate(() => window.runParity());
  await page.goto('http://127.0.0.1:4173/');
  await page.waitForFunction(() => document.querySelector('#documents').value.includes('Zugspitze'));
  for (const precision of ['q8', 'fp32']) {
    await page.selectOption('#precision', precision);
    await page.click('#run');
    await page.waitForFunction(() => document.querySelector('#status').textContent.includes('encode + score'), undefined, { timeout: 60000 });
    const first = await page.locator('#results li').first().textContent();
    if (!first.includes('Zugspitze')) throw new Error(`Wrong top result in ${precision} UI: ${first}`);
  }
  await mkdir('reports', { recursive: true });
  await page.screenshot({ path: 'reports/browser-demo.png', fullPage: true });
  if (errors.length) throw new Error(errors.join('\n'));
  const report = { passed: true, browser: await browser.version(), backend: 'WASM CPU, one thread', results, ui: 'FP32 and Q8 ranked Zugspitze first', errors };
  await writeFile('reports/browser-parity.json', JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report, null, 2));
} finally { if (browser) await browser.close(); server.kill('SIGTERM'); }
