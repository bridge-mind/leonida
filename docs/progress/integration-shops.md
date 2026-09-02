# Integration: shops requests (docs/progress/shops.md § Integration requests)

Verified with `npm run typecheck` (clean outside the in-progress mission/character paths) and `npx vitest run` (16 files, 127 tests passing).

## 1. Actors root (player / peds / pickups visible inside interiors)

- `src/core/Game.ts` — added `readonly actorsRoot = new Group();` next to `worldRoot`/`interiorRoot` and added it to the scene alongside them.
- `src/entities/Player.ts` — constructor adds `object3d` to `game.actorsRoot`; `dispose` removes from `actorsRoot`.
- `src/systems/peds/index.ts` — `new PedRenderer(game.actorsRoot)`.
- `src/systems/pickups/index.ts` — `new PickupRenderer(this.game.actorsRoot)`.
- `src/systems/peds/PedRenderer.ts` / ragdoll code — no other `worldRoot` usage; nothing else to move.
- `src/systems/shops/PedProxy.ts` — deleted (workaround no longer needed).
- `src/systems/shops/StoreRuntime.ts` — removed the `PedProxy` import, `proxies` field, the two `proxies.push` calls, the dispose loop in `unstaff`, and the per-frame `proxy.update` (the `update(_dt, playerArmed)` signature is kept for the caller in `shops/index.ts`).
- `src/world/interiors/Interiors.ts`, `CasinoInterior.ts`, `src/systems/casino/*` — read; casino guards spawn through `peds.spawn` and nothing depended on actors being hidden. `worldRoot.visible` toggling is unchanged.

## 2. Hospital fee: single owner

- `src/systems/player/index.ts` — `respawn()` now charges `P.deathCashFraction` (10 %, capped by `deathCashMax`) as `'bail'` only when `reason === 'busted'`.
- `src/systems/economy/index.ts` — the `'wasted'` fee (`EconomyConfig.hospitalFraction` 5 %, `hospitalMin` $100) fires once on `player:respawned` (emitted once per respawn from `Player.respawn`). Removed the now-dead `hospitalFrame` guard and its bookkeeping in `add()`.

## 3. Esc while a modal panel is open

- `src/systems/ui/index.ts` — already satisfied by a concurrent edit: `if (input.isBlocked) return;` precedes the Esc→`pause()` check, and the pause menu (Esc→`resume()`) and full map (Esc/M→close) handle Esc in their own branches above that gate. No change made (adding `&& !input.isBlocked` would be redundant).

## 4. `chopshop` module discovery

- `src/systems/chopshop/index.ts` — new: `export { CHOPSHOP_MODULE as default } from '../shops/ChopShop';` (id `'chopshop'` = folder name; priority `PRIORITY.robbery + 5` = 525, between shops 510 and casino 530).
- `src/systems/shops/index.ts` — removed `registerChopShop()` (call, method, `CHOPSHOP_MODULE` import). Required, not optional: `SystemRegistry.add` does not dedupe and shops (510) initialises before chopshop (525), so `systems.has('chopshop')` was false during `ShopSystem.init` and the module would have been created/initialised twice.
- `src/systems/shops/ChopShop.ts` — doc comment on `CHOPSHOP_MODULE` updated to describe discovery via the folder.

## 5. Weapon prices: prices.ts authoritative

- `src/data/weapons.ts` — synced to `WEAPON_PRICES`: smg 1200/60, shotgun 1500/80, rifle 3000/100, sniper 5000/150 (pistol 500/30 already matched). Fields kept; doc comments note prices.ts is the source of truth.
- `tests/weapons/damage.test.ts` — expected prices updated to the same numbers (test was pinning the old values).

## Not done / notes

- `src/world/interiors/ConvenienceInterior.ts:9` still has a stale comment mentioning proxy rigs on `sceneRoot`; left untouched (outside the named lines).
- `docs/progress/shops.md` "Known issues" bullets about invisible actors / PedProxy / Esc opening the pause menu are now resolved; doc not edited.
