# Agent 3 — Pedestrians + Pickups (`peds`)

## Works

- `Ped` (`src/entities/Ped.ts`) implements every `IPed` member: kinematic capsule (Groups.PED), pooled bodies created lazily, `moveToward`/`stop`/`face` primitives, `setBrain`, `applyDamage`/`kill`, `ragdoll`, `enterVehicle`/`exitVehicle`, `headHeight = 1.5`, `state` setter that resets `stateTime`.
- One shared `KinematicCharacterController` (autostep 0.4, snap to ground) steps all peds (`src/systems/peds/PedMotion.ts`). Peds within 60 m step every fixed tick; 60–150 m every third tick with 3× dt; beyond that no physics.
- Ragdoll: body → Dynamic with rotations enabled, impulse + toppling torque, freeze after 3 s (Fixed, collision groups reduced so corpses do not block cars/player), despawn 20 s later or immediately past 150 m. Alive ragdolls (vehicle bumps) stand back up after the freeze timer.
- Vehicle contact via `physics:contactStarted`: ≥ 6 m/s → `applyDamage({type:'vehicle', amount: speed*8})` + `ragdoll(vel*60 + up*150)`; slower → `knockedDown` for 2 s (fallen pose), `ped:knockedDown`, get up, brain gets `hitByVehicle` → FLEE. `KINEMATIC_KINEMATIC` active collision types enabled so kinematic traffic cars report hits.
- Instanced rendering (`PedRenderer.ts`): 7 `InstancedMesh` (head, hair, torso, arms, legs) from `pedPartGeometries()` + hair box, per-instance colors, capacity 160, animated only within 60 m, rigid rig following the dynamic body when ragdolled, hidden (scale 0) in vehicles/pooled. `computePedPartMatrices` exported from `PedBuilder.ts`; existing `buildPed`/`animatePed` API untouched.
- Spawning (`PedSpawner.ts`): N = round(80 × district.pedDensity × timeFactor) clamped 30–100 (night 0.6, Vice Beach night 1.3), spawn on `randomSidewalkPointNear(player, 40, 120)` outside the view frustum, despawn > 150 m, ~1 in 6 spawns are 2–3 ped idle groups. District/time type weights from `src/data/peds.ts`.
- `SpatialHash<Ped>` (cell 20) rebuilt per frame; `query` / `witnessesNear` fill caller arrays without allocation.
- `CivilianBrain`: WANDER along `SidewalkPolyline`s with 1–4 s pauses and crossing to another sidewalk within 25 m at ends (crosswalk preferred); IDLE_GROUP circles chatting 10–30 s; COWER on `weapon:aimedAt` within 12 m (hands up, `ped:cowered`, `ped.cower`), robbery after 1.5 s (cash pickup $10–150 in front, `crime:committed rob_ped`, then FLEE), back to WANDER 2 s after aiming stops; FLEE at 5.5 m/s along sidewalks away from the threat, re-pick every 2 s, `ped:fled`, scream, back to WANDER after 12 s and > 60 m; CALL_POLICE for 10 % of witnesses (4 s timer → `ped:reportedCrime` → FLEE); FIGHT (gang + 10 % of civilians on melee damage): approach, punch every 0.8 s within 1.2 m (`player.applyDamage 8 melee`), give up after 20 s / > 30 m; armed gang keep 8–15 m and fire via `weapons.npcFire` every 0.9 s. Reacts to `vehicle:exploded` (60 m), `vehicle:horn` (6 m behind), `crime:committed` (loud 40 m / any 25 m).
- `GangBrain`: fights back by default; 50 % of gang spawns get `weaponId: 'pistol'`.
- Carjack hook: `vehicle:carjacked` spawns a ped of `driverTypeId` at the position already fleeing.
- Damage/death: subtracts health, `ped:attacked`, `ped.hit`, blood puff via `particles.emit`, on death `ped:died`, `ped.death`, ragdoll along `info.direction`; emits `crime:committed kill_ped/kill_cop` only when `game.damage` is absent and `source === 'player'`. Dead gang peds drop $20–80 40 % of the time.
- `data/peds.ts`: `PED_TYPES`, `PED_TYPE_IDS`, `PedTypeDef` (palettes with variants, speeds, districtWeights, nightWeight, fightChance, defaultWeapon, isCop, ambient flag) for beachgoer, tourist, business, jogger, elderly, clubber, worker, gang, dealer, cop, swat, guard, clerk, lucia, jason.
- `PickupSystem` (`src/systems/pickups/`): pooled, 4 instanced meshes (cash stack, gun-shaped box, blue armor box, white box with canvas red cross), bob + spin, magnet within 2.5 m, collect within 1.2 m, 60 s lifetime with blink, cap 64, `pickup:collected`; cash → `economy.add(value,'pickup')`, health → `player.heal`, armor → `player.addArmor`, weapon → `weapons.give`, ammo → `weapons.addAmmo`.
- Autotests in `src/debug/autotests/peds.ts` (5 scenarios, see below). `npm run typecheck`: zero errors in owned files.

## Cut or deferred

- No separate crouch pose for armed gang peds; they use the existing `aiming` arm pose while standing.
- Peds do not path around each other (the character controller slides them past; no steering/avoidance).
- No per-ped LOD mesh; distant peds are simply not animated (static pose), which is cheap enough at ≤ 160 instances.
- Horn reaction is a short FLEE (1.5 s of "step aside" by fleeing the vehicle position), not a sidestep animation.
- Idle-group chatter has no gestures/audio beyond the standing pose.
- `Ped.ts` is ~470 lines (motion moved to `PedMotion.ts`); further splitting would scatter entity state.

## Known bugs

- A ped spawned exactly on a road edge can pick a sidewalk on the far side of the street and cross mid-block (no crosswalk there) when no other sidewalk end is within 25 m.
- Ragdoll capsules occasionally jitter on top of a moving vehicle before freezing.
- If `world` finishes loading after `peds` (async city build), the first spawn wave is delayed until `roadGraph` exists; the autotest allows 8 s for this.
- Fleeing peds can run through building doorway volumes if the interiors system marks them as sensors (we exclude sensors in the controller filter by design).

## How to test

- Isolated run: `?systems=world,player,camera,economy,peds,pickups`
- Autotests: append `&autotest=peds` (runs all five scenarios sequentially; logs `AUTOTEST PASS peds` / `AUTOTEST FAIL peds <reason>`). Scenarios: crowd ≥ 40 peds within 8 s with no console errors; `spawnHere:business` + `aimAt:<id>` → `cower` within 1 s, cash pickup within 3 s, scripted KeyW collects it and `economy.cash` rises (skipped without economy); `gunshot` → ≥ 50 % of peds within 40 m in `flee`/`callPolice` within 3 s and a `ped:reportedCrime` within 6 s; `spawnHere:worker` + vehicle damage 90 + ragdoll → `dead` and despawned (uses `peds.fastForward`); 50 wandering peds ≥ 90 % within 3 m of `nearestSidewalkPoint`.
- Keys: WASD walk, hold right mouse / aim (via weapons system) at a ped for the cower/robbery loop; bump peds with a car (traffic system) for knockdown/ragdoll.
- Debug actions (`&actions=` or `game.debug.run('...')`):
  - `peds.spawn:<type>:<x>:<z>` → id, `peds.spawnHere:<type>` (3 m ahead, facing the player) → id
  - `peds.count`, `peds.stats` (counts per state + target), `peds.state:<id>`
  - `peds.aimAt[:<id>]` (simulated `weapon:aimedAt` for 2 s), `peds.gunshot` (fake loud `shot_fired` at the player)
  - `peds.ragdollNearest`, `peds.killNearest`, `peds.ragdoll:<id>`, `peds.damage:<id>:<amount>[:<type>]`
  - `peds.setDensity:<n>` (override N; `peds.setDensity:x` clears), `peds.fastForward:<seconds>` (advance ragdoll timers)
  - `pickups.spawn:<kind>:<value>[:<weaponId>]` (2 m ahead), `pickups.count`, `pickups.clear`

## Integration requests

1. `src/core/Events.ts` (optional, nice-to-have): the ped system currently declares `ped:attackedPlayer` and `ped:droppedCash` via declaration merging in `src/systems/peds/events.ts`. If the UI/audio engineers want to consume them without importing our module for types, move these two lines into the shared `GameEvents`:

```ts
/** A ped punched or shot at the player (for UI hit direction, audio). */
'ped:attackedPlayer': { ped: IPed; melee: boolean };
/** A gang ped or victim dropped cash on death/robbery. */
'ped:droppedCash': { ped: IPed; amount: number };
```

2. `src/core/entities.ts` (optional): `PedContext` gains `peds`, `rng`, `isNight` via merging in our `events.ts`. Brains written by other agents get them for free, but hoisting them into the core interface documents them for everyone:

```ts
peds: IPedsPeer;
rng: Random;
isNight: boolean;
```

3. Vehicles engineer: kinematic traffic cars only trigger `physics:contactStarted` against peds because we enabled `ActiveCollisionTypes.KINEMATIC_KINEMATIC` on the ped collider. Nothing to change unless vehicle colliders are created with `events: false`; they must have contact events enabled.

## Public API

- `game.peds` (`PedSystem implements IPedsPeer`): `spawn(typeId, position, { brain?, heading?, weaponId? }) → IPed`, `despawn(ped)`, `query(pos, radius, out, filter?)`, `witnessesNear(pos, radius, out)`, `byId(id)`, `nearest(pos, filter?)`, `live: readonly Ped[]`, `typeIds`, `debugActions`.
- `game.pickups` (`PickupSystem implements IPickupsPeer`): `spawn(kind, position, value, weaponId?) → Entity`, `despawn(pickup)`, `query(pos, radius, out)`, `live`, `debugActions`.
- `Ped` (`IPed` + `IDamageable`): `health`, `velocity`, `weaponId`, `handsUp`, `isCop`, `vehicle`, `state`, `stateTime`, `sinceDamage`, `brain`, `heading`, `typeId`, `type: PedTypeDef`, `headHeight`, `isAlive`, `ragdolled`, `setBrain`, `moveToward(pos, speed)`, `stop()`, `face(pos)`, `forward(out)`, `ragdoll(impulse)`, `knockDown(source, impulse)`, `enterVehicle(vehicle, seat)`, `exitVehicle()`, `kill(source)`, `applyDamage(info)`, `notify(event)`.
- Brains: `CivilianBrain`, `GangBrain`, `SidewalkNav` from `src/systems/peds/brains`; events for custom brains in `src/systems/peds/events.ts` (`BrainEvent`, `asBrainEvent`, `toPedEvent`, `damagedEvent`).
- `src/gfx/PedBuilder.ts`: unchanged `buildPed`, `animatePed`, `PedPalette`, `PedPose`, `PedRig`, `PED_DIMS`, `PED_PALETTES`, `pedPartGeometries`, `setPedPalette`, `disposePed`; new `computePedPartMatrices(pose, heading, position, out: Matrix4[7], rotation?)`, `PED_PART_NAMES`, `PED_PART_COUNT`, `PED_HAIR`.
- `src/data/peds.ts`: `PED_TYPES`, `PED_TYPE_IDS`, `PedTypeId`, `PedTypeDef`.
- Tunables: `src/systems/peds/config.ts` (`PedConfig`), `src/systems/pickups/config.ts` (`PickupConfig`).
