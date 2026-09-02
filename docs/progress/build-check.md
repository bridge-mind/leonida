# Build check — 2026-09-02

Scope: `npm run build` (= `tsc --noEmit && vite build`) and `npm run typecheck`, static inspection of `dist/`. No browser, no dev server, no git, no `npm install`.

## Build result

- `npm run build`: **PASS** (exit 0). vite v8.2.2, 291 modules transformed, ~2 s.
- `npm run typecheck`: **PASS** (exit 0).
- One intermediate run failed on a concurrent mid-flight edit (`src/ui/PauseMenu.ts(37,20): TS2564 'abortButton' has no initializer` / `TS6133 'abortButton' is declared but its value is never read`). Waited 60 s, rebuilt: gone. Not a real error.

## Fixes made

- `src/debug/Debug.ts` — autotest glob changed to `import.meta.glob(['./autotests/*.ts', '!./autotests/util.ts'])`; `util.ts` is a shared helper (`Suite` class, no default export) and was being registered as a runnable autotest, so `?autotest=util` would throw instead of reporting "no such autotest". Verified: dist glob map now lists 14 suites, `util.ts` excluded.

No changes were needed to `vite.config.ts`, `tsconfig.json`, `index.html`, `package.json`, `src/main.ts` or `src/ui/styles.css`.

## Checks performed (all OK)

- `dist/index.html` references `./assets/index-<hash>.js` (module, crossorigin) and `./assets/index-<hash>.css`; `base: './'` honoured; dev `/src/...` paths rewritten.
- `@dimforge/rapier3d-compat`: WASM is inlined (base64 in `rapier.mjs`); no `.wasm` file emitted in `dist/assets`, `RAPIER.init()` in `src/main.ts` is the only entry. No plugin needed; `build.target: 'esnext'` already permits the required syntax.
- `import.meta.glob('../systems/*/index.ts')` resolves 25 system entries in the bundle, matching the 25 directories in `src/systems/`. No `tests/**` files pulled in (tests live outside `src/`, tsconfig `include` covers them for typecheck only).
- URL param parsing survives minification: `.get(\`systems\`)`, `.get(\`autotest\`)`, `.get(\`actions\`)` each present once in `dist/assets/index-*.js`; `autotest` string appears 25x, `systems` 48x.
- No dev-only URLs in `dist/assets` (`localhost`, `127.0.0.1`, `/src/` absent). No `process.env` / `require(` / `__dirname` / `global.` in `src/**/*.ts`.
- CSS: all 17 `.css` files under `src/ui/**` are imported exactly where used; `styles.css` linked once from `index.html`. Shared `menu.css` / `bottom-stack.css` / `missions.css` are imported from several TS modules — Vite dedupes them (7 CSS assets emitted, no duplicated rules). No `@import` / `url()` to missing files.
- Rollup emitted no circular-dependency or dynamic-import warnings.

## Bundle sizes (final build)

Total: 64 JS chunks + 7 CSS assets.

| Asset | Raw | gzip |
| --- | --- | --- |
| `index-*.js` (main: three + rapier + core) | 3,428.46 kB | 1,241.55 kB |
| `world-*.js` | 80.38 kB | 28.96 kB |
| `audio-*.js` | 66.10 kB | 19.26 kB |
| `casino-*.js` | 65.14 kB | 20.97 kB |
| `ui-*.js` | 41.83 kB | 13.76 kB |
| `shops-*.js` | 39.33 kB | 13.99 kB |
| `missions-*.js` | 37.82 kB | 13.39 kB |
| `peds-*.js` | 37.32 kB | 12.26 kB |
| `police-*.js` | 28.21 kB | 9.55 kB |
| `ui-*.css` | 12.37 kB | 3.07 kB |
| `index-*.css` | 1.65 kB | 0.74 kB |
| remaining 55 JS chunks | each < 22 kB | each < 8 kB |

Main chunk is 1.24 MB gzipped — under the ~2.5 MB threshold, and Vite emits no chunk-size warning (`build.chunkSizeWarningLimit: 4000` already set in `vite.config.ts`). Per the brief, no `manualChunks` split was added. Optional follow-up: a `manualChunks` for `three` and `@dimforge/rapier3d-compat` would improve long-term caching but is not required for correctness.

## Warnings remaining

- None from `tsc` or `vite build`.
- Note (not a warning): `build.chunkSizeWarningLimit: 4000` suppresses the default 500 kB advisory; the main chunk (3.4 MB raw) would otherwise trigger it. Left as-is intentionally.

## Errors you could not fix

- None. The only error observed (`src/ui/PauseMenu.ts(37,20) TS2564/TS6133 'abortButton'`) was transient from a concurrent edit and resolved on the 60 s retry.
