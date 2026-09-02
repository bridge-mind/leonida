# Agent 5 — UI + navigation

Owned: `src/systems/ui/**`, `src/systems/navigation/**`, `src/ui/**` (minus weapon wheel, crosshair, casino, phone), `src/debug/autotests/ui.ts`.
Isolation set: `?systems=world,player,camera,economy,navigation,ui&notitle`.

## Works

**NavigationSystem** (`src/systems/navigation/index.ts`, priority 700, `implements INavigationPeer`)
- One blip per landmark at init (id = landmark id, kind = landmark kind, position = door, label = name); `game.state.blips` rebuilt in place on every change.
- `setWaypoint(pos | null)` writes `game.state.waypoint` (copy) and a `waypoint` blip; A* over `roadGraph` from the player's nearest node to the target's nearest node, node positions + exact target written into `game.state.route` in place (pooled point objects, zero steady-state allocation). Leading/trailing nodes are trimmed when the player/target already sits along the first/last edge so the line does not double back.
- Re-plans when the player is > 30 m from the nearest route segment (checked every 0.5 s); auto-clears within 12 m of the waypoint; `distanceToWaypoint` measured along the remaining route (drives the minimap distance text).
- `routeTo(pos)` returns a fresh `Vector3[]`; waypoint survives save/load (`serialize/deserialize`).

**UiSystem** (`src/systems/ui/index.ts`, priority 800, `implements IUiPeer`, every member) mounts under `#ui`, calls `component.update(state, realDt)` per frame. Components (one per file in `src/ui/`, each with its own CSS):
- **HUD** (top-right): cash ticking toward the true value (rate = max(20/s, |delta|/1 s)), floating `+$1,200`/`-$500` delta (merges rapid same-sign changes, fades 2 s); six ★ slots with `.active` (red) / `.searching` (white) / `.responding` (black, white stroke) from `state.wanted`, `.flash` for 2 s on `wanted:cleared`; weapon label + `24 / 120` (hidden for fists/null, blinks while reloading); clock from `time.clockLabel`; chips while `state.interior === 'casino'`.
- **Minimap** (bottom-left 260×200, rounded clip, DPR-aware): static city pre-rendered once to an offscreen layer (`MapLayer`: water, land rects, district `mapColor` fills, roads by width/kind, landmark footprints); per frame: rotate so the player's forward is up (rotation = `heading − π`, verified against the rig convention heading 0 = +Z; north is up at heading π), zoom out with `state.vehicle.speedKmh`, route (purple), all blip kinds as glyph squares / dots / pin, waypoint + mission blips pinned to the rim, `wanted.searchCenter` flashing circle while `searching`, hot scenes, cops (blue, red when `game.wanted.state === 'active'`) and vehicles (only while wanted) through the optional peers, player arrow, `N` rim marker. Health/armor bars underneath (health blinks ≤ 25%), waypoint distance text (`350 m` / `1.2 km`).
- **FullMap** (M): full-screen canvas + legend side panel, drag to pan, wheel to zoom 0.3–4 px/m around the cursor, left click sets a waypoint, click within 20 m of the existing one clears it, Esc/M closes; coordinates/district readout under the cursor; pushes/pops the `map` input blocker and releases pointer lock.
- **Bottom stack**: objective with `~y~ ~b~ ~g~ ~r~` markup (safe DOM, no innerHTML), prompts as `[F] Enter vehicle` (rebuilt only when the joined text changes; codes like `KeyF`/`Digit1`/`Space` are shortened), progress bar (`progressBar.show(label, fraction)`).
- **Notifications** (top-left): `notify()` + `notification` event, 4 s, max 3, `money`/`warning` tints, money feed entries for `economy:cashChanged` with |delta| ≥ 100 and humanised reasons (`hospital` → Hospital bill, `bail`, `pickup`, `chopshop`, `rob_store`, …; fallback capitalises the raw reason), "Game saved" on `save:saved`.
- **Cards**: `card(kind, text, seconds)` → letterbox bars, huge display text (WASTED red, BUSTED blue, MISSION PASSED gold gradient, MISSION FAILED red, custom white; a non-matching `text` becomes a subtitle), `#game` gets `.ui-desaturate`, `time.scale = 0.25` while showing and back to 1 only if still 0.25; resolves after fade-out. Auto: `player:died` → WASTED, `player:busted` → BUSTED. `letterbox(on)` toggles the bars alone.
- **Dialogue**: bottom-center subtitles, speaker color hashed from the name (Lucia pink, Jason cyan), per-line time `seconds ?? clamp(len × 0.06, 1.8, 6)`, Space advances while active and not paused, `hide()` cancels, `active` getter, `dialogue.tick` per line.
- **PauseMenu** (Esc): Resume, Map (full map on top), Stats (cash, earned/spent, kills, shots fired, missions passed, heat from `game.wanted`, time played, day; refreshed every 0.5 s), Settings (mouse sensitivity → `Config.camera.mouseSensitivity` clamped 0.0008–0.005, master/sfx/music → `audio.setBusVolume`, aim assist + shadows toggles; shadows toggle applies to the renderer live; all persisted as JSON at `localStorage['leonida.settings']`), Save, Load, Controls table (§18), Quit to title. Pausing saves/zeroes `time.scale`, stops the clock, pushes the `pause` blocker, releases pointer lock, emits `ui:paused`/`ui:resumed`. Keyboard (↑↓/WS + Enter) and mouse. Because browsers swallow Esc while the pointer is locked, losing pointer lock (not initiated by the UI) also opens the pause menu.
- **TitleScreen**: shown at boot unless `notitle`, `systems`, `autotest` or `actions` is in the URL. LEONIDA neon logo, New Game / Continue (disabled without a save), controls footer; camera in cinematic orbit over Vice Beach (`orbit({700,0,0}, 260, 70, 0.04)`) while the simulation runs; `title` blocker; New Game → `cinematic.stop()`, third-person (or vehicle) camera, snap, `ui:newGame`, `ui.title.stinger`; Continue → `save.load()` + `ui:continue`. Quit-to-title re-shows it.
- `isModal` = title ∨ pause ∨ full map. `prompt(key, text)` / `clearPrompt(key)` manage UI-owned prompt objects in `state.prompts`. P quick-saves when no menu is open.
- HUD-family overlays receive `dt = 0` while paused and fade out (`.menu-open`) when a menu owns the screen.

## Cut or deferred
- Static map layer is rendered at **1 px/m** (tunable `UiConfig.map.staticPxPerMeter`) instead of 0.5 px/m: the full map zooms to 4 px/m and 0.5 px/m looked blurry there. Memory: one 2000×2000 canvas.
- Invert-Y, draw distance and bloom settings are omitted (the camera/renderer expose no hooks). Aim assist and shadows are stored for others to read.
- No "Recalculating…" text (per spec). No 3D route rendering.
- Dialogue does not block input, so Space also jumps while a line is showing (spec accepted this).
- The card sound/`wanted.lost` jingle for cleared stars is left to the wanted system; the HUD only flashes.
- Blips are not clustered/de-duplicated on the full map; mission/contact labels appear in the legend, not on the map.

## Known bugs
- `player:died` → the player's 4 s death timer runs in scaled time, so at `time.scale = 0.25` the WASTED sequence takes ≈7 real seconds (4 s slow-mo card, then ≈3 s at full speed before the hospital respawn). Fix belongs to the player system (see integration requests).
- If a browser delivers the Esc keydown *and* drops pointer lock in the same instant, a 0.3 s grace window prevents an immediate resume; if it delivers Esc later than that the menu closes again (not observed in Chromium).
- `debug.runActionString` splits on `:`, so `ui.objective:` texts cannot contain colons from the URL; use `game.debug.run('ui.objective', text)` from the console instead.
- Not verified in a browser this session (browser automation was forbidden); only `npm run typecheck` was run. Layout at 1440×900 / 1920×1080 still needs the screenshot review from the integrator.

## How to test
- Typecheck: `npm run typecheck` (zero errors in owned files).
- Isolation run: `npm run dev` → `http://localhost:<port>/?systems=world,player,camera,economy,navigation,ui&notitle`; drop `&notitle` to see the title screen.
- Autotest: `npm run smoke -- --systems=world,player,camera,economy,navigation,ui --name=ui --autotest=ui` (expects `AUTOTEST PASS ui/*` for: systems-present, waypoint-route, landmark-blips, minimap-renders, waypoint-clears-on-arrival, wanted-star-classes, notification-shows-and-expires, money-notification-for-cash-change, card-slow-mo, objective-markup, prompts-render, progress-bar, dialogue-advances-with-space, pause-resume, full-map-toggle, title-screen-flow, no-errors-after-run).
- Keys: `M` full map (drag / wheel / click), `Esc` pause, `Space` advance dialogue, `P` quick save, `Enter`/arrows in menus.
- Debug actions (`game.debug.run(...)` or `&actions=`): `ui.notify:<text>:<kind>`, `ui.card:<kind>:<text>[:<seconds>]`, `ui.dialogue`, `ui.objective:<text>`, `ui.progress:<fraction>` (negative hides), `ui.title`, `ui.pause`, `ui.resume`, `ui.map`, `ui.state`, `navigation.waypoint:<x>:<z>`, `navigation.clear`, `navigation.route`, `navigation.blips`.
- Star states: `game.state.wanted.level = 3; game.state.wanted.state = 'responding'` (or `searching` / `active`) from the console; `game.events.emit('wanted:cleared', {reason:'x'})` for the flash.

## Integration requests
1. **`src/entities/Player.ts` (foundation)** — run the death timer on real time so the WASTED card and the respawn line up regardless of slow-mo:
   ```ts
   // in Player.fixedUpdate (or wherever `this.deathTimer -= dt` lives)
   this.deathTimer -= this.game.time.scale > 0 ? dt / this.game.time.scale : dt;
   ```
   Why: `ui.card('wasted')` sets `time.scale = 0.25` for 4 real seconds; with scaled dt the respawn happens ~3 s after the card is gone.
2. **`src/core/Input.ts` (foundation)** — the canvas `mousedown` handler calls `c.requestPointerLock()` without handling the returned promise; in Chromium a rejected request (e.g. right after Esc) surfaces as an unhandled rejection and is counted as a console error by `Debug`. Suggested:
   ```ts
   if (!this.pointerLocked && this.enabled) void Promise.resolve(c.requestPointerLock()).catch(() => {});
   ```
3. **Wanted system (Agent 8)** — emit `wanted:cleared` in the same frame `state.wanted.level` drops to 0 (or one frame before) so the HUD flashes the stars that were lit; play `wanted.lost` there (the HUD does not).
4. **Missions / robbery / shops** — please use `game.ui.prompt(key, text)` / `clearPrompt(key)` for UI-owned prompts, or push your own `{key, text}` object into `game.state.prompts` exactly like `InteriorsSystem` does (keep the same object instance so removal by identity works).

## Public API
```ts
// game.ui (IUiPeer)
notify(text: string, kind?: 'info' | 'money' | 'warning'): void
card(kind: 'wasted' | 'busted' | 'missionPassed' | 'missionFailed' | 'custom', text: string, seconds = 4): Promise<void>
dialogue.show(lines: { speaker: string; text: string; seconds?: number }[]): Promise<void>
dialogue.hide(): void
dialogue.active: boolean
prompt(key: string, text: string): void            // key is a KeyboardEvent code ('KeyF', 'KeyE', 'Space')
clearPrompt(key: string): void
setObjective(text: string | null): void            // supports ~y~ ~b~ ~g~ ~r~ markup
progressBar.show(label: string, fraction: number): void
progressBar.hide(): void
letterbox(on: boolean): void
pause(): void; resume(): void; isPaused: boolean; isModal: boolean
// events: 'ui:newGame', 'ui:continue', 'ui:paused', 'ui:resumed'

// game.navigation (INavigationPeer) + extras on NavigationSystem
setWaypoint(pos: V3 | null): void
waypoint: V3 | null
addBlip(blip: { id; kind: BlipKind; position: V3; label? }): void   // position is kept by reference (live markers move)
removeBlip(id: string): void
routeTo(pos: V3): Vector3[]                        // fresh array; empty when unreachable
distanceToWaypoint: number                         // meters along the route (NavigationSystem only)

// Settings for other systems: JSON at localStorage['leonida.settings']
{ mouseSensitivity, masterVolume, sfxVolume, musicVolume, aimAssist: boolean, shadows: boolean }
// or import { loadSettings } from 'src/systems/ui/settings'
```
