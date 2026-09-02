# Agent 8 — Wanted & Police (`wanted`, `police`)

Files: `src/systems/wanted/**`, `src/systems/police/**`, `src/data/police.ts`, `src/debug/autotests/{wanted,police}.ts`, `tests/wanted/machine.test.ts`.

## What works

### WantedSystem (`src/systems/wanted`, `PRIORITY.wanted = 400`, `implements IWantedPeer`)

- `data/police.ts` — `CRIME_TABLE` (stars from 0 / stars when already wanted / witness rule / repeat cooldown / hot-scene flag) for every `CrimeType`: shot_fired +1 (witnessed), kill_ped +1 / +2 while wanted, kill_cop +2, carjack +1 (witnessed), steal_police_vehicle +2, rob_ped +1 (witnessed), rob_store +2, hit_cop_vehicle +1, hit_cop +1, explosion +2, attack_helicopter +1, kill_swat +1. Cap `MAX_WANTED_LEVEL = 6`.
- `crimes.ts` — listens to `crime:committed` (actor `player` | `companion`), `ped:reportedCrime` (phone call → at least level 1, `responding`, search center at the crime), `vehicle:collided` (player car ↔ cruiser or cop above 3 impulse → `hit_cop_vehicle` / `hit_cop`, 8 s cooldown), `damage:applied` (player bullet/melee on a living cop → `hit_cop`; lethal hits are left to the damage system's `kill_cop`), `vehicle:entered` (player into a police car → `steal_police_vehicle`). `reportCrime(type, position, actor)` for direct callers.
- Witnesses (`witnessesOf`): a cop within 40 m (60 m loud) sees the crime directly → the level rises and the state goes straight to `active` (no phone call, sight clock started). Otherwise `peds.witnessesNear(pos, loud ? 60 : 25)` non-empty, or already wanted, counts as witnessed. Unwitnessed `civilian`-rule crimes are ignored.
- `machine.ts` — pure state machine (unit-tested): `none → responding → active ↔ searching → none`. `active` while a cop has had LOS in the last 1 s; 3 s without LOS → `searching` with `searchCenter` = last seen position and `searchRadius` 40 m + 8 m/s up to 220 m + 30 m × level. Evasion timer = level × 20 s + 20 s, running only while unseen; outside the circle it runs at full speed, inside (hidden) at 1/1.5. Any counted crime while wanted resets the timer and may add stars; re-acquired LOS → `active` and the timer resets.
- Hot scenes: every hot-scene crime records `{position, radius: 60, expiresAt}` (merged within 30 m, max 8, 120 s expiry). While at level 0, re-entering a live scene re-raises to level 1 `searching`. Published to `state.wanted.hotScenes`.
- Heat 0..100: +5 per star gained, −1/min decay, exposed as `heat`; read by the police for response delay and accuracy. Only heat is serialized; a loaded game starts at level 0.
- `clear(reason)`: level 0, state `none`, `wanted:cleared {reason}` emitted in the same frame after `state.wanted` has been written (HUD flash sees the lit stars); `wanted.lost` plays for `evaded` / `respray`. Clears on `player:respawned`. `wanted:changed {level, state, previousLevel}` on every level or state change; `wanted.star.gain` on star up, `wanted.searching` on entering search, `wanted.hotscene` on re-raise.
- `vision.ts` — cop line of sight: every 0.25 s, up to 6 rays round-robin over the cops in range (foot 60 m, in a cruiser 90 m; helicopters have their own 250 m check and report through the private `wanted:sighting` event), `physics.raycast` eye → player chest (or car roof) with groups STATIC | VEHICLE excluding the player's own body/car. A cruiser within 6 m counts as seeing without a ray. `playerSeen` = any hit within the last 1 s.
- `game.state.wanted` is written every fixed step: `{level, state, searchCenter (null while active / not wanted), searchRadius, hotScenes}`.

### PoliceSystem (`src/systems/police`, `PRIORITY.police = 410`, `implements IPolicePeer`)

- `DISPATCH_TABLE` (data/police.ts) by level: 1 → 1 cruiser (2 cops, pistols, arrest); 2 → 2 cruisers, ramming; 3 → 3 cruisers + roadblock with 2 foot cops, SMGs, lethal; 4 → + SWAT van (4 swat, rifles) + helicopter, shotguns; 5 → 4 cruisers, 2 vans, 2 helis, roadblocks with spikes, 3 cops each; 6 → 3 vans, rifles everywhere, +0.2 accuracy. Caps: 12 units / 28 cops (config).
- `dispatch.ts` — keeps the tier's unit counts alive every 1.5 s after a response delay of 4 s (→ 1 s at max heat). Spawn points: road-graph lane points 150–250 m from the player, outside the camera frustum, ≥ 8 m from other vehicles; `vehicles.spawn('police' | 'swat', ...)` + `setDriver(new PoliceDriver)` + `setSiren(true)`; crews via `peds.spawn('cop' | 'swat', ..., {brain, weaponId})` then `enterVehicle`. Cops are held in `state = 'scripted'` so the ambient ped spawner never recycles them. Units farther than 320 m and off-screen are recycled. The player carjacking a cruiser drops it from the fleet (crew bails, driver removed).
- `PoliceDriver.ts` (`VehicleDriver`) — A\* route (`graph.findPath`) to the player's nearest node, re-path every 2 s, lane-offset node following like `TrafficDriver`; within 40 m with LOS: steer at the predicted player-car position and ram at full throttle (tiers ≥ 2), otherwise trail 9 m behind; player on foot within 25 m: brake, crew exits; stuck detection (< 0.6 m/s for 2 s → reverse 1.3 s); stand-down mode drives away along the route / current heading. Destroyed cruiser → crew bails.
- `CopBrain.ts` / `SwatBrain.ts` (`PedBrain`) — ride → approach (run > 15 m, walk closer) → arrest (≤ 2.5 m with LOS, `arrestContact`) at levels 1–2 while the player is unarmed and not fleeing; otherwise combat: keep a 12–20 m standoff, strafe with random switch timing, back off when crowded, chase past 30 m, hold fire without LOS, `weapons.npcFire(ped, weapon, eye, dir, accuracy)` at `COP_FIRE_INTERVAL[weapon]` with jitter, accuracy 0.35 + heat/300 + tier bonus (SWAT 0.6, always lethal). Being shot / rammed (`onEvent` damaged / hitByVehicle) makes the cop lethal ("provoked"). Knockdowns are left to the ped's own timer.
- `arrest.ts` — level ≤ 3, player on foot standing still (or in a car under 1 m/s) with any cop in arrest contact for 1.0 s continuous (timer decays at 2× instead of snapping) → `Freeze!` prompt, then BUSTED: leaves the car, `game.player.kill('busted')` (player system respawns at the nearest police station and charges the bail via `economy.trySpend(…, 'bail')`), emits `player:busted` (UI card + `wanted.busted` sting), `wanted.clear('busted')`.
- `Helicopter.ts` — own `Entity` + `IDamageable` (600 HP): Three.js group (body, tail, boom, skids, spinning main/tail rotors), kinematic Rapier body with a cuboid collider registered as the damageable so hitscan bullets hurt it; orbits the player at 45 m altitude / 35 m radius, night SpotLight aimed at the player, gunner fires a rifle every 0.4 s at accuracy 0.3 with a 250 m LOS check that also feeds `wanted:sighting`; `vehicle.engine.helicopter.loop` positional loop. Player hits raise `attack_helicopter` (6 s cooldown); at 0 HP it falls for 3 s, `damage.explode`s (radius 9, 220 dmg) and is removed. Stand-down flies off before removal.
- `Roadblock.ts` — road edge 120–200 m ahead along the player's heading (`planRoadblock`), two cruisers parked across the lanes (no driver, sirens on), 2–3 foot cops standing behind them, optional spike strip (thin box). Player car crossing the strip above 4 m/s: `applyDamage` 150 + lateral impulse + `vehicle.crash.medium`, 3 s cooldown. Rebuilt when the player passes it or gets 320 m away, 25 s cooldown.
- `IPolicePeer`: `activeUnits` (cruisers + vans + helis + roadblocks), `copsNear(pos, r, out)` (foot + in cruisers) for the minimap.
- Stand-down on `wanted:cleared`: sirens off, drivers leave, cops walk; units removed after 8 s once off-screen or 20 s hard cap; everything removed immediately on `player:respawned`.
- Runs with only core + world + player + camera loaded: wanted logic still works, police spawns nothing.

## How to test

- `npm run typecheck` — zero errors in the owned files.
- `npx vitest run tests/wanted` — 15 pure tests on `WantedMachine` (crime table, witness rules, cooldowns, responding/active/searching/evaded transitions, search-circle growth, hidden-inside multiplier, hot-scene re-raise, heat decay, cap).
- Autotests (`?autotest=`): `wanted` → witnessed-shot-raises-level-1-responding, cop-line-of-sight-makes-active, losing-sight-enters-searching-then-evades, hot-scene-reentry-reraises, phone-report-raises-responding, crime-while-wanted-adds-stars-and-caps, respawn-clears-level. `police` → level-4-dispatches-swat-and-helicopter, clear-stands-units-down, helicopter-takes-damage-and-crashes, roadblock-spawns-across-the-road-ahead, level-1-cop-adjacent-busts-standing-player, no-console-errors. Both need `world,player,camera,peds,vehicles,traffic,weapons,damage,wanted,police` (ui optional). Not run in a browser this session (forbidden); only typecheck + vitest were executed.
- Debug actions (`game.debug.run(...)` / console):
  - `wanted.set(level, state?)`, `wanted.clear()`, `wanted.state()` → `{level, state, heat, seen, evasion, evasionSeconds, searchRadius, hotScenes}`, `wanted.crime(type, loud?)` (emits `crime:committed` at the player), `wanted.fastForward(seconds)` (advances the evasion timer), `wanted.seen()`, `wanted.heat(value)`.
  - `police.spawnCruiser('police' | 'swat')` (40 m behind, chasing), `police.spawnCop(x?, z?, weapon?)`, `police.units()` → `{cruisers, swat, helicopters, roadblocks, cops, footCops, arrestTimer}`, `police.roadblock(spikes?)`, `police.heli()`, `police.clear()`.

## Integration requests

1. **`src/systems/ui/index.ts` (UI engineer)** — the Wasted sting/card flashes for one frame before Busted. `Arrest.bust()` calls `player.kill('busted')` (→ `player:died`) and then emits `player:busted` in the same call stack; the Cards component plays a sting on every `show`. Deferring the Wasted card by a microtask lets the Busted card win cleanly:

   ```ts
   // src/systems/ui/index.ts, in init(), replace the player:died subscription:
   game.events.on('player:died', () => {
     // BUSTED arrives synchronously after player:died; let it claim the card first.
     queueMicrotask(() => {
       if (this.cards.activeKind !== 'busted') void this.card('wasted', 'Wasted');
     });
   }),
   ```

   ```ts
   // src/ui/Cards.ts, next to `get isActive()`:
   get activeKind(): CardKind | null {
     return this.active?.kind ?? null;
   }
   ```

2. **`src/core/Events.ts` (optional, integrator)** — `player:died` carries no reason. Adding `reason?: 'wasted' | 'busted'` (set by `Player.kill`) would make request 1 unnecessary and let the audio/UI skip the Wasted sting outright. Not required for this stage.

3. **`src/systems/vehicles` (vehicles engineer, optional)** — no tire/max-speed API exists, so spike strips use `vehicle.applyDamage({amount: 150, type: 'collision'})` plus a lateral impulse. A `vehicle.setSpeedLimit(factor)` or `puncture()` hook would make spikes feel right (car limps instead of being damaged).

No other files need changes; every peer is consumed through the optional `game.<peer>` getters.

## Known issues

- Wasted card/sting shows for a frame before Busted until request 1 is applied (Cards replaces the active card, so the end state is correct).
- Cruisers do not perform a true PIT: ramming aims at the predicted player-car position; with high grip the cruiser can bounce off rather than spin the target.
- Roadblock placement probes a single point 120–200 m ahead along the player's heading and takes the nearest lane point within 60 m; when the road bends away or the player faces open ground/water, `planRoadblock` returns null and the dispatcher retries on the next tick (25 s cooldown after a successful block).
- Helicopter gunner and spotlight ignore interiors: while `state.interior !== null` the LOS ray still runs against exterior geometry, so a player inside an interior with an open sky is treated as hidden only if the interior has a roof collider.
- Cops in a cruiser share the driver's LOS ray for `wanted:sighting`; foot cops each get their own throttled ray, so an unusually large crowd (> 24 foot cops) needs more than a second to sweep every cop once.
- `steal_police_vehicle` also fires when the player re-enters a cruiser they already own (5 s cooldown limits it to once per hijack).
- Autotests were only typechecked this session; timing tolerances (10 s for level-4 dispatch, 20 s for the helicopter crash) were chosen conservatively and may need tightening after a browser run.

## Cut scope

- No distinct "army" archetype at level 6: it reuses `swat` vans/peds with rifles and a +0.2 accuracy bonus, as allowed by the brief.
- No cover-point system: cops "take cover" by holding a 12–20 m standoff and strafing, not by using geometry.
- No player-car speed reduction from spikes (see integration request 3).
- Helicopters do not drop SWAT ropes; roadblocks are static (cops do not re-enter the parked cruisers to chase).
- No serialization of active units or level; only heat survives save/load.

## Performance notes

- No per-frame allocations in the hot paths: shared scratch `Vector3`s, reused query arrays (`IPed[]`, `IVehicle[]`), `Map<IPed, CopBrain>` populated only on spawn/despawn, hot-scene records reused by `publish()`.
- Raycasts are throttled: wanted vision 6 rays / 0.25 s max, each `CopBrain` 1 ray / 0.3 s, `PoliceDriver` 1 ray / 0.25 s, helicopter 1 ray / 0.25 s; A\* only every 2 s per cruiser (immediately on stand-down).
- Dispatch runs at 1.5 s intervals; unit audit (dead / destroyed / player-stolen) every 1 s; roadblock planning is one `nearestLanePoint` query.
- Helicopter is a single Three.js group (7 meshes) + one kinematic body; the SpotLight is only enabled at night (`time.isNight`) and off while standing down.
- All unit counts are capped by `PoliceConfig.dispatch.maxUnits` / `maxCops`; units off-screen beyond 320 m are recycled instead of accumulating.
