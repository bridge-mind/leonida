# Agent 4 — Weapons, Damage & FX (`weapons`)

## Works

- `src/data/weapons.ts`: `WEAPONS: Record<WeaponId, WeaponDef>` straight from SYSTEM_PROMPT §8 (fists, pistol, smg, shotgun, rifle, sniper — damage, rpm, magazine, range, spread degrees, price, ammo price per mag, reload seconds 1.5–2.5, shotgun `pellets: 8`, `automatic` for smg/rifle), display `name` + HUD `label`, sound ids (`weapon.<id>.fire/reload/empty`, fists → `weapon.melee.*`). `WEAPON_ORDER` (slots 1–6), `isWeaponId`, `shotInterval`.
- `WeaponSystem` (`src/systems/weapons/`, `implements IWeaponsPeer`, `PRIORITY.weapons`):
  - `Inventory.ts`: `Map<WeaponId, {mag, reserve}>`, starts with fists (`WeaponsConfig.startWithPistol` default false). `give` (adds ammo when already owned, auto-equips when unarmed), `addAmmo`, reload maths, `cycle` (wheel scroll skips unowned), `serialize`/`deserialize`.
  - Player input in `fixedUpdate` only while alive, `!input.isBlocked`, `!ui?.isModal`. Fire edge for semi-auto, held for automatic, rate-limited by rpm; `R` reload with table timer (`weapon:reloaded` + sound; auto-reload on dry mag); `1–6` quick select (owned only); `wheelDelta` cycles; hold `Tab` opens the wheel. In a vehicle only pistol/smg fire (along the camera forward from the window; melee/aim disabled). `game.state.weapon` published every frame.
  - `Hitscan.ts`: origin `camera.three.position`, direction = camera world direction (or toward `input.mouseWorld` when set — first pellet exact, extra pellets spread, `exactScriptedAim`), random cone spread (× `hipFireSpreadMul` when not aiming), `physics.raycastInto` on STATIC|VEHICLE|PED excluding `vehicles?.playerVehicle ?? player.entity`. Body part by `headHeight` (default 1.5). Damage via `game.damage.apply`, falling back to `applyDamage` when the damage system is absent. Impact FX per `StaticEntity.surface` (concrete/asphalt/wood/metal/dirt/sand/grass/water/glass), peds → blood + `impact.flesh`, vehicles → sparks + `impact.vehicle`, dummies → concrete + `impact.wood`. Tracer muzzle→hit (or max range), `muzzleFlash` particles + one shared 40 ms `PointLight`, recoil (`camera.pitch -= recoil`, `fovKick(1.2)`, `shake(0.03, 0.08)`), `weapon:fired`, `crime:committed shot_fired` (loud), `stats.shotsFired++`, empty click once per press.
  - `Aim.ts`: while aiming, spread-free raycast every frame; ped/vehicle/dummy within 40 m → `weapon:aimedAt {target, seconds}` (accumulated, reset on target change); `aimTarget` exposed. Aim assist (`aimAssist`, default true): on the frame aiming starts, nearest ped/dummy inside a 10° cone within 40 m → one `camera.setOrbit` snap to chest height. Sniper: `camera.fovKick(-25)` per aiming frame + `.scope` overlay in `#ui`.
  - `Melee.ts`: fists + LMB on foot: 3-hit combo (0.35 s spacing, 1 s reset), 1.5 m reach, target = nearest ped from `peds.query(pos, 2)` passing a facing-dot test, else a short raycast (dummies/props). 15 melee damage through `game.damage`, third hit `ragdoll(forward*180 + up*60)` on peds; `weapon.melee.swing/hit/knockdown`, blood puff, small camera shake.
  - `npcFire(shooter, weaponId, origin, direction, accuracy)`: spread × (1 + 3(1 − accuracy)), deliberate miss chance (1 − accuracy) × 0.5 with a 0.12 rad offset, raycast excluding the shooter, player hits go through `damage.apply` → `player.applyDamage`, positioned fire sound, tracer/flash at origin, `weapon:fired {shooter}`.
  - `Dummy.ts`: `TargetDummy` (`kind: 'prop'`, `headHeight: 1.5`, 0.8×1.8×0.5 box, static cuboid on `Groups.PED`, health 100, turns red per hit, black + collider disabled when dead). `spawnDummy`/`clearDummies`/`dummies`.
  - `serialize()` → `{inventory: {slots, current}}`; `deserialize` restores inventory and equips.
- `DamageSystem` (`src/systems/damage/`, `implements IDamagePeer`, `PRIORITY.damage`): `apply` (headshot ×3 for peds/dummies, player routed to `player.applyDamage`, `isDamageable` otherwise), `damage:applied`, `weapon:hit {target, killed}` for player-sourced hits, `entity:killed` once per entity id, `stats.kills` + `crime:committed kill_ped/kill_cop` for player kills of peds. `explode`: peds via `peds.query`, vehicles via `vehicles.query`, player by distance, linear falloff with full damage inside radius/3, ped ragdoll impulse, vehicle `applyDamage({type:'explosion'})`, `particles.emit('explosion', {scale: radius/6})`, `explosion.large`, distance-scaled `camera.shake`, `crime:committed explosion` when player-sourced.
- `ParticleSystem` (`src/systems/particles/`, `implements IParticlesPeer`, `PRIORITY.particles`): two `THREE.Points` pools (additive 2048 for sparks/flash/fire/explosion/cash, normal 2048 for dust/smoke/blood/water/glass) with a custom `ShaderMaterial` (per-particle size/color/alpha/shape: soft circle, square, streak), `depthWrite: false`, swap-remove, zero allocations after init. `TracerBuffer` = pooled `LineSegments` (128, 60 ms fade). Every `ParticleEffectId` is mapped in `effects.ts`; `tracer` needs `opts.end`, `muzzleFlash` also drives the shared `MuzzleFlash` sprite + light from `src/gfx/Effects.ts`. `opts.scale` scales count/size/speed, `opts.color` tints, `opts.direction` orients bursts. Particles falling below `floorY` are recycled.
- UI: `WeaponWheel` (radial 6-slot overlay, `time.scale = 0.2` while open, restores only if still 0.2, mouse-delta selection, `ui.wheel.open/tick/select`, equips owned weapon on release, blocks input while open) and `Crosshair` (visible while aiming or driving with pistol/smg; hit marker white / red on kill with `ui.hitmarker[.kill]`; reload arc under the crosshair; sniper scope overlay).
- Pure tests `tests/weapons/*.test.ts` (spread cone, NPC spread, falloff, weapon table, inventory) pass; `npm run typecheck` has zero errors in owned files.

## Cut or deferred

- No visual arm swing for melee / no gun meshes on the player rig (player rig is owned by another agent); a camera nudge and sounds stand in.
- Vehicle drive-by uses the camera forward directly, no window-side restriction or passenger aiming.
- No weapon pickups/shop hooks here — `give`/`addAmmo` exist for pickups/economy to call.
- Interiors: weapons are allowed everywhere; robbery reactions live in other systems.
- Particle rendering is unsorted, single-draw-call per pool; no soft particles or lighting.

## Known bugs

- Ped hits produce two blood puffs (Hitscan emits one, `Ped.applyDamage` also emits one). Harmless visually; either side can drop theirs once the integrator decides.
- `WeaponSystem.index.ts` is 404 lines (debug actions already split to `debug.ts`).
- The `weapon:aimedAt` raycast and the fire raycast both run per frame while aiming and firing (two rays); acceptable cost.
- Sniper `fovKick(-25)` fights `Config.camera.aim.fov` only through the per-frame kick; releasing aim snaps back over the camera's own recovery time.

## How to test

- Isolated run: `?systems=world,player,camera,weapons,damage,particles`
- Autotests: `&autotest=weapons` (default export in `src/debug/autotests/weapons.ts`, driven only via `game.debug.run` and `game.input` scripted methods). Scenarios: give/equip pistol; body shot → dummy 75; headshot → dead (`entity:killed`); reload timing ±10 % of table; SMG held 2 s → 24–30 `weapon:fired`; shotgun single `weapon:fired`, ≥ 6 `damage:applied` with differing hit points; `damage.explode` 3 m vs 12 m from two dummies; Tab wheel visible + `time.scale === 0.2`, release restores and equips the selected slot; quick-select digits (unowned ignored); `weapon:aimedAt` on a dummy; melee 3-hit combo on a dummy; `npcFire` at the player lowers health; particle pools bounded; `shot_fired` crimes == shots fired; no console errors.
- Keys: `RMB` aim, `LMB` fire / punch, `R` reload, `1–6` quick select, mouse wheel cycle, hold `Tab` wheel.
- Debug actions: `weapons.give:<id>:<ammo>`, `weapons.equip:<id>`, `weapons.fire`, `weapons.reload`, `weapons.state`, `weapons.spawnDummy:<x>:<z>`, `weapons.dummies`, `weapons.clearDummies`, `weapons.aimAssist[:true|false]`, `damage.explode:<x>:<z>:<radius>:<damage>`, `particles.emit:<id>:<x>:<z>`, `particles.count`, `particles.clear`.
- Unit tests: `npx vitest run tests/weapons`.

## Integration requests

- None required. Everything is wired through existing `game.<peer>` getters and declared events.
- Optional (peds owner): drop the blood emit in `Ped.applyDamage` when `game.weapons` is present, or tell me to drop mine, to avoid the double puff.

## Public API

- `game.weapons` (`IWeaponsPeer`): `current`, `isAiming`, `aimTarget`, `isReloading`, `inventory()`, `give(id, ammo?)`, `equip(id)`, `addAmmo(id, amount)`, `npcFire(shooter, weaponId, origin, direction, accuracy)`. Extra: `aimSeconds`, `dummies`, `spawnDummy(x, z)`, `clearDummies()`, `fireOnce()`, `startReload()`.
- `game.damage` (`IDamagePeer`): `apply(target, info)`, `explode(position, radius, damage, impulse, source)`.
- `game.particles` (`IParticlesPeer`): `emit(id, position, opts?)` with `opts.end` (tracer), `opts.direction`, `opts.scale`, `opts.color`.
- Data: `WEAPONS`, `WEAPON_ORDER`, `WeaponDef`, `isWeaponId`, `shotInterval` from `src/data/weapons.ts`.
- Events emitted: `weapon:fired`, `weapon:hit`, `weapon:reloaded`, `weapon:aimedAt`, `damage:applied`, `entity:killed`, `crime:committed` (`shot_fired`, `kill_ped`, `kill_cop`, `explosion`). Private (`src/systems/weapons/events.ts`): `weapon:wheelToggled {open}`, `weapon:meleeSwung {combo, hit}`.
- State: `game.state.weapon = {id, ammoInMag, ammoReserve, isReloading}`; `stats.shotsFired`, `stats.kills`.
- Gfx: `src/gfx/Effects.ts` exports `particleSpriteTexture`, `particlePointMaterial`, `flashTexture`, `MuzzleFlash`.
