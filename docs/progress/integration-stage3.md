# Integration: Stage 3 (wanted / police / shops / robbery / economy / interiors) against Stage 2

Read-only review of the Stage 3 folders against `src/core/*`, the Stage 2 systems they call, the autotests and the priority bands. Edits were confined to Stage 3 folders, `src/ui/phone`, `src/ui/shops` and `src/debug/autotests/*`. `npm run typecheck` and `npx vitest run` (16 files, 127 tests) pass after the edits. No browser runs were made.

## Fixed

- `src/systems/shops/brains/GuardBrain.ts` — guards listened for `heardGunshot`, which nothing emits; the ped system broadcasts gunfire as `crimeSeen { loud: true }`. Guards now go hostile on a loud `crimeSeen` (and on `explosion`), so shooting inside the jewelry store / bank actually draws fire.
- `src/ui/phone/Phone.ts` — consumed `ArrowUp` / `Escape` / `Backspace` after handling them. Shops (510) run before UI (800): closing the phone popped its input blocker and the same Escape edge then opened the pause menu in the same frame.
- `src/ui/shops/GunStore.ts` — consumed `Backspace` / `KeyE` / `Escape` on close. `ShopSystem.update()` calls `gunStore.update()` and then `updateCounter()` in the same frame, so leaving with E re-opened the panel immediately (it could never be closed with E); Escape also fell through to the pause menu.
- `src/systems/wanted/crimes.ts`, `src/systems/wanted/config.ts` — `witnessesOf` used the 2-D ped spatial hash. Interiors live at `Config.interiors.origin` (0, 600, 0), directly above the city, so a shot fired inside a store was "witnessed" by street peds 600 m below. Added `witness.maxVerticalGap` (12 m) and filter witnesses (cops and civilians) by |Δy|.
- `src/systems/police/index.ts` — subscribed to `vehicle:carjacked`. `PlayerControl.completeEnter` emits it *before* `v.enter(player, 0)`; a police cruiser still had its cop in seat 0, so the player was seated in a rear seat (`playerDriven` false, no controls, no way out). The crew now bails synchronously in the handler and the unit is dropped, so seat 0 is free when the player mounts.
- `src/systems/shops/index.ts` — `refreshBlips()` skipped for the `chopshop` store. `ChopShopSystem` subscribes to `game:ready` / `time:dayChanged` first and writes `Chop Shop — wants <archetype>`; shops then overwrote it with the plain name every time.
- `src/debug/autotests/robbery.ts` — `respray-clears-wanted` used `wanted.setLevel(2)`, which defaults to `active` and marks the player seen, so the check always hit the "refused while seen" early-out and never tested the clear. Now `setLevel(2, 'searching')`. `--systems` list in the header now includes `chopshop`.
- `src/debug/autotests/shops.ts` — `--systems` list includes `chopshop` (it is a discovered module now, not registered by shops); assertion message updated.

(The duplicate-`chopshop` registration — `ShopSystem.registerChopShop()` in `init()` plus discovery of the new `src/systems/chopshop/index.ts`, which would have created two instances and a double payout on E — was removed concurrently by the integration-shops engineer while this review was running; verified gone.)

## Verified OK

1. **Module registration** — `wanted` 400, `police` 410, `economy` 500, `shops` 510, `chopshop` 525 (via `src/systems/chopshop/index.ts` → `CHOPSHOP_MODULE`), `robbery` 520, `interiors` 540; ids equal folder names. `Game.boot` creates+inits modules sequentially in priority order, then emits `game:ready`; shops defers interior registration and blips to `game:ready` (`game.interiors` is `undefined` during `init`, guarded), chopshop defers its blip to `game:ready`, robbery reads shops lazily via `systems.get`. Nothing reads `navigation`/`ui` in `init()`.
2. **Crime → stars** — every Stage 2 emitter's `CrimeType` is in `CRIME_TABLE` (`shot_fired`, `kill_ped`, `kill_cop`, `carjack`, `steal_police_vehicle`, `rob_ped`, `rob_store`, `hit_cop_vehicle`, `hit_cop`, `explosion`, `attack_helicopter`, `kill_swat`); `rob_store` is `witness: 'none'` so casino/robbery hold-ups always count. `ped:reportedCrime` → `reportedByPhone`; cop witnesses mark seen → `active`. `wanted:changed {level, state, previousLevel}`; `game.state.wanted.{level,state,searchCenter,searchRadius,hotScenes}` published every fixed step and on change, matching `HUD.ts` / `MapDraw.ts`; states `'none'|'responding'|'searching'|'active'`. `wanted:cleared` emitted from the machine's `cleared` hook in the same call that drops the level; `player:respawned` → `clear('respawned')`. Respray clears only when `!wanted.playerSeen` (`Respray.spray`).
3. **Police ↔ peds/vehicles** — `PoliceDriver.update(vehicle, dt)` matches `VehicleDriver`; writes `throttle/brake ∈ [0,1]`, `steer ∈ [-1,1]` with the same sign convention as `TrafficDriver` (`-err·gain`), `handbrake` boolean; reversing via `brake` at low speed matches `handling.ts`. Sirens via `setSiren`. Cops spawn as `'cop'`/`'swat'` (both in `src/data/peds.ts`, `isCop: true`) with tier weapon ids; cruiser seats 0..1 (4-seat `police`), van seats 0..3 (6-seat `swat`); crews exit with `exitVehicle()`. `CopBrain`/`SwatBrain` use `update(ped, ctx, dt)` / `onEvent`. LOS through `physics.raycastInto` with `STATIC|VEHICLE` and `exclude: target`. `npcFire(shooter, weaponId, origin, dir, accuracy)` with a normalised direction from eye height. Arrest → `game.player.kill('busted')` + one `player:busted`; bail is billed only by the player system (`respawn('busted')`), police never touches economy. Helicopter and roadblock use existing groups; roadblock cruisers are `parked` and skipped on dispose when the player took one. `wanted:cleared` → `standDownAll`, sweep + hard cap despawn; `maxUnits`/`maxCops` respected. `copsNear(pos, radius, out)` returns live `IPed`s (2-D), which is what `Minimap` draws.
4. **Shops/robbery ↔ peds/weapons/interiors** — `ClerkBrain`/`GuardBrain` cast through `asBrainEvent`; `aimedAt`/`damaged` shapes match. Robbery compares `weapon:aimedAt.target` by object identity with the `IPed` returned by `peds.spawn` (same instance). `weapons.current`, `give`, `addAmmo` signatures match (`Purchases.ts`). Cash pickup: `worldPoint(layout.register)` is origin + slot-offset local point; `pickups.spawn` snaps to `groundHeightAt` (counter top) + hover, horizontal collect/magnet radii (1.2 / 2.5 m) reach it from the counter position. `progressBar.show(label, fraction)` / `hide()`. Closed state serialised as `{closed: {id: untilDay}}` and re-applied after late interior registration. `interior:entered/exited {id}` match `Interiors.ts`. Blip kinds come from `landmark.kind` (already `BlipKind`).
5. **Chop shop / respray** — `vehicles.playerVehicle`, `vehicles.despawn` (force-exits the player), `Vehicle.color` is a writable field read directly by `VehicleRenderer.write` each frame (no cache). Valuation: `state.health / archetype.health` × `archetype.chopValue`; police vehicles refused.
6. **Phone** — `Input.moveAxis` is WASD-only; `ArrowUp` is not a movement key, so the "nudges the player" note in shops.md is stale. Snapmatic uses `game.renderer` / `game.scene` / `game.viewCamera` (real Game members). Contacts use `game.missions?.available()` and `start(id)`.
7. **Economy** — `fee()` → `add()` → exactly one `economy:cashChanged`. `state.stats` is `Record<string, number>`; economy writes `cashEarned/cashSpent/moneyFromRobberies/moneyFromChopShop`. Save shape `{cash, earned, spent, fromRobberies?, fromChopShop?}` is a superset of `{cash, earned, spent}`; `deserialize` tolerates missing fields. Hospital fee (5 %, min $100) on `player:respawned 'wasted'` only; bail only from the player system.
8. **Interiors** — 9 store interiors register once (guarded by `stores.size > 0`); every `StoreDef.landmarkId` / `id` matches a landmark `id` / `interiorId` in `src/data/landmarks.ts`. Door prompt refuses closed stores with a notification + `store.closed`; `fadeThrough` shared by respray and chop shop. Interior `build()` runs once; no per-frame work in interiors.
9. **Autotests** — all debug action names used by `wanted.ts`, `police.ts`, `shops.ts`, `robbery.ts` exist (`wanted.set/clear/state/fastForward`, `police.spawnCop/units/heli/roadblock/clear`, `shops.status/buy/open/closeUi/food/close/reopen`, `phone.open/snap/app/feed/close`, `respray.spray/position`, `chopshop.position/setWanted/sell`, `robbery.status/skip`, `economy.get/set`, `player.armor/damage/kill/teleport`, `pickups.count`, `vehicles.spawn/enterNearest`, `interiors.enter/exit`, `peds.spawnHere`) and payload shapes match. Timing of the `wanted.ts` LOS and evasion checks is consistent with `WantedConfig` (`loseSightAfter` 3 s, evasion 20 + 20·level).
10. `npm run typecheck` exit 0; `npx vitest run` 127/127.

## Integration requests

### `src/systems/vehicles/PlayerControl.ts` — do not synthesise a driver ped when the cruiser has a real crew

`completeEnter` emits `vehicle:carjacked` with `driverTypeId ?? (isPolice ? 'cop' : 'business')`; the ped system responds by spawning a *new* fleeing ped at the door. For police cruisers the driver is a real `IPed` in seat 0 (now ejected by the police system), so a phantom third cop (`isCop: true`, `CivilianBrain`) appears and also counts as a cop witness. Only emit the event when the vehicle had a virtual driver:

```ts
// src/systems/vehicles/PlayerControl.ts, completeEnter()
if (carjack) {
  const hadRealDriver = v.driverEntity !== null && v.driverEntity.kind === 'ped';
  v.setDriver(null);
  const driverTypeId = v.driverTypeId ?? (v.isPolice ? 'cop' : 'business');
  v.driverTypeId = null;
  const pos = { x: door.x, y: door.y, z: door.z };
  // A real ped in the seat is ejected by its owning system (police bails on this event); only
  // virtual traffic drivers need a ped conjured at the door.
  if (!hadRealDriver) g.events.emit('vehicle:carjacked', { vehicle: v, position: pos, heading: v.heading, driverTypeId });
  g.events.emit('crime:committed', { type: v.isPolice ? 'steal_police_vehicle' : 'carjack', position: pos, actor: 'player', loud: false });
}
```

If `vehicle:carjacked` must keep firing for the police bail, alternatively guard the spawn in `src/systems/peds/index.ts`:

```ts
ev.on('vehicle:carjacked', ({ vehicle, position, heading, driverTypeId }) => {
  if (vehicle.state.occupants.some((o) => o.kind === 'ped')) return; // real crew handles itself
  const ped = this.spawn(driverTypeId, position, { heading }) as Ped;
  ped.notify(toPedEvent({ type: 'carjacked', threat: this.ctx.playerPosition }));
}),
```

### `src/systems/peds/PedSpawner.ts` — no ambient spawning while the player is inside an interior

`update()` measures distance horizontally and spawns on sidewalks near the player's x/z; with the player 600 m up in a store the spawner keeps populating the street below and never despawns it (all within `despawnRadius` horizontally). Cheap gate:

```ts
update(dt: number, playerPos: Vector3): void {
  this.timer -= dt;
  if (this.timer > 0) return;
  this.timer = S.interval;
  if (this.game.state.interior !== null) return; // interiors sit 600 m above the city; nothing to populate
  ...
```

### `src/systems/peds/index.ts` — `witnessesNear` / `query` are 2-D

Same root cause as above; the wanted system now filters by |Δy| itself, but `peds.query` callers elsewhere (traffic panic, `broadcast` of `crimeSeen`) still reach across the 600 m gap. Consider a `maxDy` on `broadcast`:

```ts
private broadcast(position: V3, radius: number, make: ..., maxDy = 12): void {
  ...
  if (!ped.active || !ped.isAlive || ped.vehicle) continue;
  if (Math.abs(ped.position.y - position.y) > maxDy) continue;
```

### `docs/progress/shops.md`

"Up arrow nudges the player one frame" is stale: `Input.moveAxis` only reads WASD, and the phone now consumes `ArrowUp`.

## Open issues

- **Cop spawned by carjack** — until the PlayerControl/peds request above lands, stealing a cruiser produces one extra `cop`-typed civilian ped at the door (flees; counts as a cop witness for the `steal_police_vehicle` report). Cosmetic; the crime is `witness: 'none'` anyway.
- **Scripted peds vs. pool eviction** — `PedSystem.spawn` evicts `farthest()` at capacity (160) regardless of `state === 'scripted'`; a clerk/guard/cop could be recycled under heavy ambient load. `PoliceSystem.audit()` tolerates it; `StoreRuntime` would keep a dead `clerk` reference until exit (robbery start would then fail harmlessly). Low probability, not fixed.
- **Cruiser crew after bail** — bailed cops leave `scripted` for `wander` (`Ped.exitVehicle`), so `PedSpawner` may despawn them if the chase moves >150 m away; `audit()` removes them from `cops`. Acceptable, noted.
- **Autotest `wanted.ts` "losing-sight"** — after `wanted.set 2` the police dispatch fires ~3.4 s later; the test's `waitFor(searching, 6 s)` normally resolves at ~3 s, but a cruiser spawning 150–250 m away right at the edge could re-sight the player before `fastForward`. Would only flake, not fail deterministically; left as is.
- `src/world/interiors/ConvenienceInterior.ts:9` comment still mentions "proxy rigs" (PedProxy was deleted by the shops integration pass). Comment only.
