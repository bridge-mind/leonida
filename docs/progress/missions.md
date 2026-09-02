# Agent 10 — Missions, Story, Cutscenes, Companion, Characters (`missions`)

Systems: `missions` (600, `implements IMissionsPeer`), `cutscene` (610), `companion` (620), `characters` (630, `implements ICharactersPeer`). Every peer is consumed through `game.<peer>` / `game.systems.get(...)` and is optional; the whole stack boots with only `core + world + player + camera`.

## What works

### Objective model (`src/missions/`)
- `types.ts`: `Objective { text, blipKind, target(), start, update → running|done|failed, skip, stop }`, `MissionContext { game, missionId, reward, fail(), onCleanup(), fast }`, `MissionDef { id, name, contact, position, unlockedAfter?, reward, repeatable, build(ctx) }`. `build` runs per attempt so every objective carries fresh state.
- `objectives.ts` (pure, no peers): `goto(pos, r, text, {vehicle?, blip?})`, `enterVehicle(ref, text)`, `deliver(ref, pos, r, text)` (fails `'vehicle destroyed'` when the car is wrecked/exploded/gone), `kill(targets, text)`, `survive(seconds, text, {orKill?, showTimer?})`, `waitFor(event, predicate, text)`, `custom(spec)`, `action(fn)` (async-aware), `timed(inner, seconds, failReason)`. Helpers `forceMount`, `teleportPlayer`, `playerVehicle`.
- `objectivesWorld.ts` (optional peers, `console.warn` once and complete when missing): `escapeWanted`, `reachWanted(level)`, `robStore(storeId|null)` (adds the register cash from `robbery:completed` to `ctx.reward`; `crime:committed rob_store` fallback), `winAtCasino(amount)` (net `casino:won − casino:lost`; fails `'you went broke'` at `chips === 0 && cash < 100 && net < 0` with a taunt), `intimidateOrKill(target, 2 s, 15 m)` (uses `weapon:aimedAt`), `lootGrab(counter, 20 s, 6 m)` (ui.progressBar, pauses when out of range), `cutscene(def | () => def)`.
- `helpers.ts`: landmark lookups (`doorFront`, `contactMarker`), road/sidewalk/parking snapping through `world.roadGraph`, `spawnVehicle/spawnPed` with automatic cleanup on mission end (`persistent` opt-out), `sceneTalk/sceneOrbit` cutscene builders, `withCompanion()` (summons Jason, dismisses on cleanup).
- Brains/drivers: `HostileBrain` (approach + `weapons.npcFire` with LOS), `DebtorBrain` (hands up while aimed at, flees 8 s, gets in the sedan and drives a `RouteDriver` polyline route), `RouteDriver`/`IdleDriver` (scripted `VehicleDriver`s).

### Mission runner (`src/systems/missions/`)
- `Runner.ts` `MissionRun`: sequential objectives, mission-wide `player:died → 'wasted'`, `player:busted → 'busted'`, script exceptions → `'script error'` (never crashes the loop), LIFO cleanups, `fastForward()`, `abort()`.
- `index.ts` `MissionSystem`: contact markers (`Markers.ts`: translucent yellow cylinder + `contact` blip with the contact's letter), 3 m prompt `E — Start <name>`, `start(id)` (phone/debug), objective HUD via `ui.setObjective` with `~y~` markup, GPS waypoint + `objective`-kind blip that follows moving targets, `mission:started/objective/passed/failed`. Pass: `economy.add(reward,'mission')`, `stats.missionsPassed++`, autosave `game.save.save()`, `missionPassed` card, next mission unlocks. Fail: waits for the WASTED/BUSTED respawn, `missionFailed` card, then `RetryPrompt` (Retry restarts from the beginning / Quit; MenuNav + input blocker + pointer-lock pattern). `serialize/deserialize`: completed ids + active id (restarted from the beginning on load). New game starts m1's contact marker at the prison; `available()` returns `MissionSummary[]` (jobs appear after m8).
- Debug actions: `missions.list`, `missions.start(id)`, `missions.abort`, `missions.fastForward`, `missions.complete(id)` (fast mode: cutscenes/waits collapse, card seconds shortened, resolves with the final status), `missions.unlockAll`, `missions.retry('retry'|'quit')`, `missions.retryVisible`, `missions.objective`, `missions.reset`.

### Story (`src/missions/m1..m8`, `jobs.ts`; data in `src/data/story.ts`, `src/data/dialogue.ts`)
1. **Release** — orbit cutscene at the penitentiary gate (3 lines), Jason's muscle car nearby, Jason summoned as companion, drive to the safehouse following GPS, safehouse intro cutscene. $500.
2. **Gas Money** — safehouse dialogue, drive to the Port Gellhorn gas station with Jason, `robStore('gas_station')`, `escapeWanted`. Reward = register cash + $300.
3. **Repo** — Manny Ochoa at the Docks; a parked `sports` at the Vice Beach hotel lot (blip), carjack (`IdleDriver` + `driverTypeId` so PlayerControl's carjack logic applies), `escapeWanted`, deliver to Gellhorn Salvage, car despawned. $1,500.
4. **Bad Debt** — debtor in Little Habana (`DebtorBrain`), aim 2 s within 15 m → $800, or kill → $1,200; end dialogue branches on the outcome.
5. **Mud Club** — package Downtown, deliver to the Gator Pit, ambush of 6 `gang` peds (pistol/SMG `HostileBrain`), rifle pickup (90 rounds), `survive(90 s)` or kill all, escape in a spawned `pickup`. $2,000.
6. **High Roller** — Leonida Palms entrance, orbit cutscene with Dominic Reyes, `winAtCasino(2000)`. $2,500.
7. **Heat** — Downtown, `reachWanted(4)`, `survive(180 s)`, `escapeWanted`. Reward = $1,000 × peak stars (tracked from `wanted:changed`).
8. **The Score** — safehouse cutscene with Jason, Vice Rocks Jewelers (`interior:entered 'jewelry'`), 20 s loot grab at the counter (interior guards are used when present, otherwise two `guard` peds with pistols are spawned), exit → `wanted.setLevel(5)`, `sports` getaway parked at the marina, 5-minute `timed(goto marina)` (`'the boat left without you'`), end cutscene, `card('custom', 'THE END — LEONIDA')`, $25,000.
- Jobs (repeatable, unlocked by m8, listed by `available()` for the phone): `job_chopshop` (steal the chop shop's `wantedArchetype` via `IChopShopPeer`, else any `sports`; pays `max($1,500, peer.valueOf(v))`), `job_robbery` (rob any store + `escapeWanted`).

### Cutscenes (`src/systems/cutscene/`)
`play(def, {fast?})` with steps `{camera}`, `{orbit}`, `{dialogue}`, `{wait}`, `{run}`. Letterbox, `ui.dialogue.show`, input blocker, camera `cinematic` mode + `moveTo/orbit`, mode restored afterwards. Space skips the rest of the scene after a short grace; `run` hooks always execute (also in fast mode) so spawns stay consistent.

### Companion (`src/systems/companion/`)
`summon('jason'|'lucia')` spawns the matching ped type (`src/data/peds.ts` has both), `dismiss()` despawns. `CompanionBrain`: follows at 3 m (runs beyond 6 m, teleport catch-up when far), enters seat 1 when the player mounts and exits with them, shoots hostiles within 30 m (cops while wanted, `ped:attackedPlayer` memory, gang peds in fight state) with `weapons.npcFire` at 2 shots/s and LOS, fires at pursuing cops from the passenger seat. Cannot die: at `health <= 0` health is pinned to 1, `downed` for 10 s, then revives. Re-spawned if the peds system culls him.

### Characters (`src/systems/characters/`, UI in `src/ui/missions/`)
Q opens `CharacterWheel` (slow-mo 0.2, two portraits, mouse selects, input blocker + pointer lock like WeaponWheel). `switchTo(id)`: stores `{position, heading, health, armor, weapons.inventory()}` for the outgoing character, sky-up 0.6 s, teleport to the stored position (default: safehouse door), `setPedPalette(player.entity.rig, PED_PALETTES[id])`, stats (Lucia sprint 1.08; Jason recoil 0.8), weapons restored, sky-down 0.6 s, `game.state.activeCharacter`, `character:switched`. Blocked (notify) while a companion mission is active. Adrenaline (Caps Lock): `time.scale = 0.3` for 4 real seconds, vignette overlay (`AbilityFx`), `ability:activated/ended`; Jason marks ped heads within 40 m (≤ 20 DOM markers), Lucia gets `damageMultiplier 2` until her next shot. Recharge 60 s; `game.state.abilityCharge` published; serialize both states + active id + charge.

## How to test

- `npm run typecheck` — 0 errors in owned files (0 overall at the time of writing).
- `npx vitest run tests/missions` — `tests/missions/objectives.test.ts` (15 tests, fake game in `tests/missions/fakeGame.ts`): goto/deliver/enterVehicle, wasted/busted/timed/script-error fail conditions, sequencing + stop order, fastForward walk-through, abort cleanups, peer-less degradation, `robStore` reward, `winAtCasino` net/broke.
- Autotests (not run this session — browser automation was forbidden):
  - `npm run smoke -- --systems=world,player,camera,ui,vehicles,peds,weapons,damage,pickups,navigation,economy,wanted,police,missions,cutscene,companion,characters --name=missions --autotest=missions` — list/unlocks, m1 via `fastForward` to `mission:passed` (cash delta), m2 death → `mission:failed 'wasted'` + retry prompt → quit clears the blocker, `missions.complete` for m2…m8 + both jobs with zero console errors, jobs available after the finale, no leaked blocker/time scale/objective.
  - `... --systems=world,player,camera,ui,weapons,characters --name=characters --autotest=characters` — switch to Jason (event, state, camera mode restored, `rig.palette`, stats, moved), switch back, adrenaline (`time.scale ≈ 0.3` then 1, charge drops, `state.abilityCharge`), recharge, no errors.
- Manual: `game.debug.run('missions.unlockAll')`, walk into a yellow contact cylinder, E. `missions.fastForward` skips the current objective. Q = character wheel, Caps Lock = adrenaline.

## Integration requests

1. **HUD ability meter** — `src/ui/HUD.ts`: render `game.state.abilityCharge` (0..1) as a thin bar under the health/armor bars; the field is already published every frame by `CharacterSystem`. Suggested: a `div.hud-ability` whose `style.width = `${Math.round(state.abilityCharge * 100)}%``, coloured `var(--accent)` when full.
2. **Pause menu "Abort mission"** (optional) — `src/ui/PauseMenu.ts`: add an item visible when `game.missions?.active` is non-null:
   ```ts
   { label: 'Abort mission', action: () => { game.missions?.abort(); this.close(); } }
   ```
3. **Smoke system list** — whoever owns the smoke presets: add `missions,cutscene,companion,characters` to the default `--systems` set so the autotests above are picked up by the nightly run.

No changes to core contracts were needed: `IMissionsPeer`, `ICharactersPeer`, all `mission:*/character:*/ability:*` events, `GameState.objective/activeCharacter/abilityCharge`, blip kinds `contact/objective/vehicle/casino/jewelry/marina/chopshop/club`, ped types `lucia/jason/gang/guard`, store id `gas_station`, interior id `jewelry` all already exist.

## Known issues

- Autotests were written against the debug API but not executed (no browser this session); expect small timing adjustments (e.g. the retry-prompt wait after WASTED) on first run.
- m8 counter position is inferred as 5.5 m ahead of the interior entry point rather than read from the jewelry interior layout; if the shops engineer exposes a counter position, `counterFrom` in `m8_theScore.ts` should use it.
- m5's package is a marker on the sidewalk (no pickup entity — `pickups` has no cash-package kind).
- The companion enters the passenger seat only when the car has a free seat 1; in a two-seat `sports` with a scripted driver he stays outside and catches up on foot/teleport.
- `winAtCasino` counts any casino game (blackjack, roulette, slots), not blackjack only — accepted per the brief ("any table counts").
- Character switch during a vehicle ride dismounts the player first; the vehicle is left where it is (not stored per character).
- `MissionSystem` (`index.ts`) is ~405 lines including debug actions.

## Cut scope

- No mission title card at start (the objective line + `mission:started` are enough; the passed/failed cards exist).
- No boat archetype: the m8 getaway is a parked `sports` at the marina.
- Jason's adrenaline uses DOM head markers, not particles; no audio pitch shift (no bus-pitch API).
- No per-mission music/radio changes.
