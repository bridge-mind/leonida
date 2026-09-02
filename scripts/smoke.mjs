#!/usr/bin/env node
/**
 * Smoke test: starts Vite on a free port, opens the game in headless Chromium,
 * waits for __GAME_READY__, runs for --seconds, records console/page errors,
 * saves docs/screenshots/<name>.png + .log and exits non-zero on any error or AUTOTEST FAIL.
 *
 *   npm run smoke -- --name=foundation --systems=world,player,camera --autotest=foundation --actions=time.set:22 --seconds=10
 */
import { createServer as createNetServer } from 'node:net';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');

function parseArgs(argv) {
  const out = {};
  for (const arg of argv) {
    if (!arg.startsWith('--')) continue;
    const eq = arg.indexOf('=');
    if (eq < 0) out[arg.slice(2)] = 'true';
    else out[arg.slice(2, eq)] = arg.slice(eq + 1);
  }
  return out;
}

function freePort() {
  return new Promise((resolvePort, reject) => {
    const srv = createNetServer();
    srv.unref();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      srv.close(() => resolvePort(port));
    });
  });
}

async function launchBrowser(extraArgs) {
  const args = [
    '--enable-gpu',
    '--ignore-gpu-blocklist',
    '--enable-webgl',
    '--enable-unsafe-webgpu',
    '--use-angle=metal',
    '--enable-features=Vulkan',
    '--disable-gpu-driver-bug-workarounds',
    '--autoplay-policy=no-user-gesture-required',
    '--mute-audio',
    ...extraArgs,
  ];
  // `channel: 'chromium'` selects the new headless mode (full Chromium) instead of headless-shell.
  return chromium.launch({ headless: true, channel: 'chromium', args });
}

async function webglAvailable(browser) {
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  const result = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (!gl) return { ok: false, renderer: 'none' };
    const info = gl.getExtension('WEBGL_debug_renderer_info');
    const renderer = info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : 'unknown';
    return { ok: true, renderer: String(renderer) };
  });
  await ctx.close();
  return result;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const name = args.name ?? 'smoke';
  const seconds = Number(args.seconds ?? 10);
  const width = Number(args.width ?? 1440);
  const height = Number(args.height ?? 900);
  const outDir = resolve(root, 'docs/screenshots');
  mkdirSync(outDir, { recursive: true });

  const port = await freePort();
  const server = await createServer({ root, configFile: resolve(root, 'vite.config.ts'), server: { port, strictPort: true, host: '127.0.0.1' }, logLevel: 'error' });
  await server.listen();
  const base = `http://127.0.0.1:${port}/`;

  const params = new URLSearchParams();
  if (args.systems) params.set('systems', args.systems);
  if (args.autotest) params.set('autotest', args.autotest);
  if (args.actions) params.set('actions', args.actions);
  if (args.spawn) params.set('spawn', args.spawn);
  if (args.perf) params.set('perf', '1');
  const url = `${base}?${params.toString()}`;

  const log = [];
  const consoleErrors = [];
  const consoleWarnings = [];
  const pageErrors = [];
  let autotestFailed = false;
  const line = (s) => {
    log.push(s);
    process.stdout.write(s + '\n');
  };

  let browser = await launchBrowser([]);
  let gl = await webglAvailable(browser);
  if (!gl.ok) {
    await browser.close();
    line('[smoke] GPU WebGL unavailable, retrying with SwiftShader (software rendering, slow)');
    browser = await launchBrowser(['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']);
    gl = await webglAvailable(browser);
  }
  if (!gl.ok) {
    await browser.close();
    await server.close();
    console.error('[smoke] WebGL is not available in headless Chromium on this machine. Run the game in a real browser (npm run dev) instead.');
    process.exit(2);
  }
  line(`[smoke] WebGL renderer: ${gl.renderer}`);
  line(`[smoke] url: ${url}`);

  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  page.on('console', (msg) => {
    const text = msg.text();
    const type = msg.type();
    line(`[console.${type}] ${text}`);
    if (type === 'error') consoleErrors.push(text);
    if (type === 'warning') consoleWarnings.push(text);
    if (text.startsWith('AUTOTEST FAIL')) autotestFailed = true;
  });
  page.on('pageerror', (err) => {
    line(`[pageerror] ${err.message}`);
    pageErrors.push(err.message);
  });

  let exitCode = 0;
  try {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.waitForFunction(() => window.__GAME_READY__ === true, null, { timeout: 90000 });
    line('[smoke] game ready');
    const deadline = Date.now() + seconds * 1000;
    // Autotests may take longer than --seconds; keep waiting while one is pending.
    while (Date.now() < deadline || (await page.evaluate(() => window.__game?.debug?.autotestResult === 'pending'))) {
      await page.waitForTimeout(250);
      if (Date.now() > deadline + 120000) {
        line('[smoke] autotest still pending after 120 s past deadline; giving up');
        autotestFailed = true;
        break;
      }
    }
    const perf = await page.evaluate(() => JSON.stringify(window.__game.perf.toJSON()));
    line(`[smoke] perf ${perf}`);
    const systems = await page.evaluate(() => JSON.stringify(window.__game.debug.run('core.systems')));
    line(`[smoke] systems ${systems}`);
    if (args.dump) {
      // --dump=a.b,c.d prints the result of debug actions at the end of the run.
      for (const action of args.dump.split(',')) {
        const result = await page.evaluate((a) => {
          try {
            return JSON.stringify(window.__game.debug.runActionString(a)[0]);
          } catch (err) {
            return `ERROR ${err instanceof Error ? err.message : String(err)}`;
          }
        }, action.trim());
        line(`[smoke] dump ${action.trim()} = ${result}`);
      }
    }
    await page.screenshot({ path: resolve(outDir, `${name}.png`) });
    line(`[smoke] screenshot docs/screenshots/${name}.png`);
  } catch (err) {
    line(`[smoke] ERROR ${err instanceof Error ? err.stack ?? err.message : String(err)}`);
    exitCode = 1;
    try {
      await page.screenshot({ path: resolve(outDir, `${name}.png`) });
    } catch {
      /* page may be gone */
    }
  }

  if (consoleErrors.length || pageErrors.length) exitCode = 1;
  if (autotestFailed) exitCode = 1;
  line(`[smoke] console errors: ${consoleErrors.length}, warnings: ${consoleWarnings.length}, page errors: ${pageErrors.length}, autotest failed: ${autotestFailed}`);
  line(`[smoke] result: ${exitCode === 0 ? 'PASS' : 'FAIL'}`);
  writeFileSync(resolve(outDir, `${name}.log`), log.join('\n') + '\n');

  await context.close();
  await browser.close();
  await server.close();
  process.exit(exitCode);
}

main().catch((err) => {
  console.error('[smoke] fatal', err);
  process.exit(1);
});
