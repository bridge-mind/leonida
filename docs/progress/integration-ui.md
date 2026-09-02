# Integration — UI requests (wanted / missions / characters)

Files touched: `src/ui/Cards.ts`, `src/systems/ui/index.ts`, `src/ui/HUD.ts`, `src/ui/hud.css`, `src/ui/PauseMenu.ts`.
Verified: `npm run typecheck` (0 errors), `npx vitest run` (16 files / 127 tests pass). No browser run.

## Fixed

1. **BUSTED wins over WASTED** (wanted.md request 1). `Player.kill` emits `player:died` with `{ position }` only (no reason), and `src/entities/Player.ts` is out of scope, so the microtask route was taken:
   - `Cards.activeKind: CardKind | null` added next to `isActive`.
   - `UiSystem` now defers the auto Wasted card with `queueMicrotask` and skips it when `cards.activeKind === 'busted'`. `Arrest.bust()` emits `player:busted` synchronously in the same call stack as `kill('busted')`, so the Busted card is already active when the microtask runs: no Wasted card, no `wanted.wasted` sting; `wanted.busted` plays once (only `Cards.show` plays stings, and it is called once).
   - Plain deaths are unchanged apart from a one-microtask delay (same frame).
2. **HUD ability meter** (missions.md request 1). `HUD` renders `state.abilityCharge` as a thin 120×6 px bar (`div.hud-ability` → `.hud-ability-bar` → `.hud-ability-fill`, `transform: scaleX`) with a muted "ABILITY" label, in the top-right HUD cluster under the clock, in the health/armor bar style (same track colours, `--color-accent-yellow` fill). `.full` adds a subtle glow and tints the label. Only built when `game.systems.has('characters')`; DOM is written only when the rounded percentage changes (`lastAbilityPct`).
3. **Pause menu "Abort mission"** (missions.md request 2). New `menu-button` between Controls and Quit to title; `game.missions?.abort()` → `resume()` → `relockPointer()` (same pattern as Resume). Shown only while `game.missions?.active` is non-null: `refreshAbort()` runs on `show()` and every frame while the menu is visible, toggling both `disabled` (so `MenuNav` skips it with the keyboard) and inline `display: none` (`.menu-button` is `display: block`, which would override the `hidden` attribute, and `pause-menu.css` was not editable).

## Open

- The bar sits in the HUD (top-right) rather than literally under the health/armor bars, which belong to `Minimap`/`minimap.css` (not in the editable set). If the UI owner prefers it under the minimap, move the `hud-ability*` markup/CSS into `Minimap` after the bars row.
- wanted.md request 2 (`reason?: 'wasted' | 'busted'` on `player:died`) was not taken: the emitter in `Player.ts` would need to set it and is out of scope. It would let the audio `player.death` cue and `MissionRunner` (which currently fails with `'wasted'` on `player:died` before `player:busted` arrives) branch on the reason as well.
- Not verified in a browser (forbidden this session): the abort button layout with 9 items in `.pause-buttons` and the ability bar at 1440×900 need a screenshot check.
