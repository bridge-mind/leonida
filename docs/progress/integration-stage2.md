# Integration pass — Stage 2 (audio, casino, peds, ui, vehicles, weapons)

Read-only review of the completed Stage 2 systems against `docs/TEAM-PLAN.md` §3 and
`SYSTEM_PROMPT.md` §19–§24, with surgical fixes. No browser; `npm run typecheck` (0 errors
project-wide at time of writing) and `npx vitest run` (16 files / 127 tests) pass.

## Fixed

- `src/systems/vehicles/PlayerControl.ts` — player who dies while mounted (shot in the seat, `player.kill('busted')`) is now dismounted at once; previously `Player.fixedUpdate` returned early while mounted so `deathTimer` never ran and the player stayed dead forever (camera returns to `thirdPerson` automatically via the on-foot target).
- `src/core/Input.ts` — added `unlockRequested` (set by `exitPointerLock()`, cleared on relock) so the UI can tell a deliberate pointer release from the user pressing Esc; added `consume(code)` to drop a press edge handled by raw DOM listeners.
- `src/systems/ui/index.ts` — `onPointerLockChange` honours `input.unlockRequested`: opening a casino panel (or any future modal that calls `input.exitPointerLock()`) no longer triggers the pause menu.
- `src/systems/ui/index.ts` — Esc / M / P are ignored while a foreign blocker is active (weapon wheel, casino panel, phone, shop); Esc in a casino panel used to close the panel *and* pause. Dialogue `Space` still advances.
- `src/systems/ui/index.ts` — pausing with the weapon wheel open left `time.scale` at 0.2 after resume (pause → pointer unlock → `releaseAll()` releases Tab → wheel hides while scale is 0 and cannot undo its slow-mo → `resume()` restores the saved 0.2). `weapon:wheelToggled {open:false}` while paused now resets the saved scale to 1.
- `src/systems/ui/index.ts` — pointer is re-locked after Esc-resume, after closing the full map (Esc/M), and `quitToTitle` shows the title before `resume()` so no relock flicker.
- `src/ui/casino/Panel.ts` — Esc consumes the input edge; `close()` re-requests pointer lock when no UI modal/pause is active (previously the player had to click the canvas again).
- `src/systems/casino/index.ts` — removed the second `casino.ambient.loop` (audio's `AmbientManager` already plays it from `state.interior`); two beds were layered.
- `src/entities/Player.ts` — removed direct `player.jump` / `player.hurt` / `player.death` plays: audio's `footstepDriver` and `eventCues` already play them from velocity / `player:damaged` / `player:died` (every cue fired twice).
- `src/world/interiors/Interiors.ts` — removed direct `interior.enter` / `interior.exit` plays (duplicated by `eventCues` on `interior:entered/exited`).
- `src/systems/player/index.ts` — `deserialize` guards `health`/`armor` (missing/NaN fields no longer poison the HUD; a save written mid-death resumes with ≥1 hp instead of 0 hp + `isAlive`).
- `src/core/Input.ts` — arrow keys removed from `moveAxis` (WASD only). `ArrowUp` is the phone key and arrows navigate menus; `docs/progress/shops.md` had logged "opening the phone nudges the player one frame".
- `src/systems/peds/index.ts` — run-over damage now goes through `game.damage.apply` with `source: 'player'` when the player is driving, so fatal run-overs emit `entity:killed` and `crime:committed kill_ped/kill_cop` like every other kill (previously silent for the wanted system). Falls back to `ped.applyDamage` in isolation runs.
- `src/systems/vehicles/VehicleFx.ts` — burning/exploding occupants (NPC drivers) are damaged through `game.damage.apply` so their deaths emit `entity:killed` once (the damage system dedupes by entity id).

## Verified OK

- All 14 completed system folders default-export a `SystemModule` whose `id` equals the folder name with a `PRIORITY.<id>` band; `pickups` exists at `src/systems/pickups/index.ts`. `Game` discovers via `import.meta.glob('../systems/*/index.ts')`, sorts by priority, and every cross-system `init()` read (`world.roadGraph`, `player.entity`) is behind `systems.has(...)`/optional peers.
- `Debug.ts` loads autotests via `import.meta.glob('./autotests/*.ts')` and registers `debugActions` for every system — new Stage 3/4 autotests load automatically.
- Event payloads: `vehicle:carjacked` → peds spawn the ejected driver at the door position (copied, not a live ref); `weapon:aimedAt` → CivilianBrain cower/drop-cash; `crime:committed` emitters (weapons `shot_fired` copies position per shot, vehicles `carjack`, damage `kill_ped`/`kill_cop`/`explosion`, casino `rob_store`); `entity:killed` emitted once per entity from `DamageSystem.apply`; `pickup:collected` routes cash→economy, health/armor→player, weapon/ammo→weapons; `vehicle:exploded` runs `damage.explode` + `traffic.panicNear`; `interior:entered/exited` set `state.interior` before emit so the audio `AmbientManager` and casino agree.
- Blood puff on ped hit is emitted once (`Hitscan.impactFx` skips peds; `Ped.applyDamage` emits).
- Blip kinds: every kind requested for Stage 3/4 (`gunstore food safehouse respray chopshop casino mission contact police hospital waypoint vehicle cop objective`) already exists in `src/ui/blipStyles.ts` with colours/glyphs, plus `store gas jewelry bank marina club prison hotScene`.
- Coordinate conventions (heading 0 → +Z, forward = (sin h, 0, cos h)) are consistent in peds spawn/face, casino guard `npcFire` (eye + normalised dir), vehicle seat/door math, pickups debug spawn.
- Key map: F enter/exit, E interiors/casino stations, Tab wheel, M map, Esc pause, R reload, H horn, L lights, Space handbrake/jump/dialogue, 1–6 weapons, P save, F3 perf — no conflicts among completed systems. Blockers are balanced: `title`, `pause`, `map`, `weaponWheel`, `casino` each push/pop on a single show/hide path (wheel also hides on `dispose`, casino panels close on interior exit/dispose).
- Time scale: `WeaponWheel.hide` and `Cards.finish` only reset `time.scale` when it still equals their own value; UI/cards/menus advance on `realDt`, HUD-family overlays freeze while paused; `Player.deathTimer` divides by `time.scale` so WASTED lines up with respawn.
- Death/respawn: single hospital/bail charge in `src/systems/player/index.ts` (economy's `hospitalFrame` guard is the only other path and it is not reached by completed code); autosave after fee + teleport; weapons cancel reload on `player:died`/`player:respawned`.
- Save/Load: `vehicles` (player car respawned beside the player), `weapons` (inventory), `navigation` (waypoint), `casino` (chips/ban), `player` (pose/health), `time` all tolerate missing fields; title "Continue" calls `save.load()` before `leaveTitle` emits `ui:continue`.
- Perf: hot loops (vehicles, traffic, peds, particles, pickups, minimap) use preallocated scratch vectors and pools; the only per-call allocations found are on rare transitions (traffic edge choice, particle burst definitions at init).
- `index.html` has `#ui` and the canvas; `main.ts` awaits Rapier before constructing `Game`; audio context is created on first gesture (`NullAudio` until then) so boot is console-clean.

## Open issues (not fixed)

1. **Remaining `damage.apply` bypasses (by design, but no `entity:killed`):** `Ped.kill()` (`src/entities/Ped.ts:233`, scripted kills), `peds.debugActions.damage` (`src/systems/peds/index.ts:394`) and `PlayerControl.exitCurrent` exit-speed damage to the player. None can produce a crime today; if a Stage 3/4 system needs `entity:killed` from scripted kills, route `Ped.kill` through `game.damage?.apply(this, …)`.
2. **Casino station `E` vs interiors exit `E`** (`src/world/interiors/CasinoInterior.ts:289`, `src/systems/interiors/index.ts`): both poll `wasPressed('KeyE')` when the player stands near a station that is also inside the exit trigger. Harmless with the current layout (nearest station is >3 m from the door). Suggested: interiors should skip the exit prompt while `state.prompts` already contains a station prompt, or casino should `input.consume('KeyE')`.
3. **`Player.deserialize` teleports before the world's ground snap** when `Continue` is pressed: `needsGroundSnap` is only set in `init()`. If a save was written while airborne the player falls a frame. Cosmetic; suggested: set `needsGroundSnap = true` in `PlayerSystem.deserialize`.

## Notes for Stage 3/4 integration

- **Pointer lock from a modal**: call `game.input.exitPointerLock()` when opening (it sets `input.unlockRequested`, so the UI system will not read the unlock as "pause"), and `game.input.requestPointerLock()` on close *only if* `game.ui?.isModal !== true && game.ui?.isPaused !== true`. Push/pop a blocker (`input.pushBlocker('<id>')`) around the modal.
- **Esc in your own modal**: if you handle Esc via a raw `keydown` listener, call `game.input.consume('Escape')` so the UI system does not also pause next frame (see `src/ui/casino/Panel.ts`). Alternatively poll `game.input.wasPressed('Escape')` in `update()` — the UI system now ignores Esc/M/P whenever `input.isBlocked`.
- **Arrow keys** are no longer movement; `ArrowUp` is free for the phone and arrows/Enter for menus (`src/ui/MenuNav.ts`).
- **Sounds**: do not play `player.hurt/death/jump`, `interior.enter/exit`, `ui.pause/unpause`, `pickup.*`, or district/interior ambient beds yourself — `src/systems/audio/eventCues.ts`, `footstepDriver.ts` and `AmbientManager.ts` do it from events/`state.interior`. Check `eventCues.ts` before adding a cue to avoid doubles.
- **Blip kinds** available (`src/ui/blipStyles.ts`): `safehouse hospital police prison casino gunstore store gas food jewelry bank respray chopshop marina club mission contact objective waypoint vehicle cop hotScene`. `mission`/`contact` use `glyph: 'label'` (first letter of the blip label); `objective`, `waypoint`, `mission` pin to the minimap rim.
- **Death while mounted**: the player is dismounted on the physics step after death; `player:dismounted` fires before `player:respawned`. Police/wanted should not re-issue BUSTED on that dismount.
- **Run-overs now count**: a fatal hit with a player-driven vehicle emits `entity:killed {source:'player'}` and `crime:committed kill_ped` (or `kill_cop` for `isCop` peds) from the damage system; the wanted system does not need its own `ped:knockedDown` → crime mapping for deaths.
- **Crime positions** in `crime:committed` are copied per emit; `ped:reportedCrime` and `vehicle:*` payloads carry entity references (live) plus copied positions — read positions from the payload, not from the entity later.
- **Pause semantics**: `ui.pause()` sets `time.scale = 0` and stops the clock; systems that keep their own slow-mo must compare `time.scale` against their own value before restoring (see `WeaponWheel.hide`, `Cards.finish`).
