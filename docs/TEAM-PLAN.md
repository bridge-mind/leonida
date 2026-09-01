# LEONIDA — Team Build Plan (10 agents)

This file is the coordination contract for the team of agents building LEONIDA. `SYSTEM_PROMPT.md` (project root) is the full game spec; `GTA6-ANALYSIS.md` is the design rationale. Read both fully before this file. Where this file and `SYSTEM_PROMPT.md` disagree on *process*, this file wins. Where they disagree on *game design*, `SYSTEM_PROMPT.md` wins.

Project root for all agents: `/Users/matthewmiller/Desktop/fable-5-1/gta6`.

---

## 1. Stages and roster

| Stage | Agent | Role | Runs |
|---|---|---|---|
| 1 | **Agent 1 — Foundation / Architect** | Phases 0–1, core architecture, contracts, tooling, smoke + autotest harness | solo |
| 2 | **Agent 2 — Vehicles & Traffic** | Phase 2 + traffic half of Phase 3 | parallel with 3–7 |
| 2 | **Agent 3 — Pedestrians** | Ped half of Phase 3 + ped robbing + cash pickups | parallel |
| 2 | **Agent 4 — Weapons, Damage & FX** | Phase 4 combat (minus wallet/HUD) + particles + weapon wheel | parallel |
| 2 | **Agent 5 — UI** | Phase 6: HUD, minimap, full map, GPS/waypoints, notifications, pause, title, cards, dialogue component | parallel |
| 2 | **Agent 6 — Casino** | Phase 8 complete (pure logic + tests + interior + UI) | parallel |
| 2 | **Agent 7 — Audio** | Full procedural audio implementation behind the audio API | parallel |
| — | **Agent 1 (resumed) — Integration pass 1** | Enable all systems together, apply integration requests, fix cross-system bugs, commit | solo |
| 3 | **Agent 8 — Wanted & Police** | Phase 5 complete | parallel with 9 |
| 3 | **Agent 9 — Economy, Shops, Robberies, Chop Shop, Phone** | Phase 7 + phone | parallel |
| 4 | **Agent 10 — Missions, Story, Cutscenes, Companion, Characters** | Phase 9 + dual protagonists + ability | solo |
| — | **Agent 1 (resumed) — Integration pass 2 / Polish / QA** | Save/load completion, performance, juice, README, final verification, commit | solo |

Stage 2 agents test in isolation against the foundation (see §3.7). Stage 3+ agents have all Stage 2 systems available and integrated.

---

## 2. Rules for every agent

1. **Read first:** `SYSTEM_PROMPT.md`, `GTA6-ANALYSIS.md`, this file, `docs/CONTRACTS.md`, `docs/ARCHITECTURE.md`, and the `src/core/` folder. Do not write code before you understand the contracts.
2. **File ownership is absolute.** Create and edit only files under your owned paths (§4). If you need a change elsewhere, write it under **Integration requests** in your progress doc with the exact file, the exact change (a diff or code block), and why. Never edit `package.json`, `src/main.ts`, `src/core/Game.ts`, `src/core/Events.ts` (use declaration merging instead), root config files, or another agent's folders. Stage 3+ agents inherit ownership of their listed paths and may *read* everything.
3. **No `npm install`.** Agent 1 preinstalls every dependency. If you need one, request it in your progress doc and implement without it.
4. **No git commands.** Agent 1 commits at integration passes. Concurrent git operations corrupt the index.
5. **No `npm run build`.** `dist/` is shared. Use `npm run typecheck`, `npm run test`, and `npm run smoke`.
6. **Typecheck discipline:** errors in files you own are yours to fix to zero. Errors in files you do not own get listed in your progress doc and left alone.
7. **Test in isolation** with `?systems=<comma-separated ids>` (§3.7). Your smoke run must pass with zero console errors and zero page errors, and you must **open and look at the screenshot PNG** it produces and confirm it shows what you built.
8. **Write autotests** in `src/debug/autotests/<yourid>*.ts` that exercise every acceptance criterion in your brief via debug actions and scripted input, printing `AUTOTEST PASS <name>` / `AUTOTEST FAIL <name> <reason>`. Your smoke run with `--autotest=<name>` must report all PASS.
9. **Progress doc:** `docs/progress/<yourid>.md` with exactly these sections: **Works**, **Cut or deferred**, **Known bugs**, **How to test** (exact commands, URL params, keys), **Integration requests**, **Public API** (what other agents can call, with signatures), **Screenshots** (paths).
10. **Quality bar is AAA feel:** tuned numbers, no jitter or popping, pooled allocations, no per-frame `new` in hot paths, strict TypeScript with no `any` outside engine boundaries, no dead code, no stubs, no `TODO` left in code. If you cut scope, say so in the progress doc.
11. **Assets are procedural only.** No network fetches at runtime or build time. No fonts, models, textures, or audio files downloaded.
12. **Time-box:** if a feature is not playable after roughly two hours of effort, ship the simplest working version, note it, and move on.
13. **Dev server:** you may run `npm run dev`; Vite picks a free port. Prefer `npm run smoke`, which starts its own server on a random port and shuts it down.
14. **Do not ask the user anything.** You have everything you need in these documents. Make sensible decisions and document them.

---

## 3. Architecture contracts (Agent 1 implements; everyone conforms)

Agent 1 writes the authoritative versions of these in `docs/CONTRACTS.md` and `docs/ARCHITECTURE.md` and in code. The shape below is binding for Agent 1.

### 3.1 System modules and discovery

- Each system lives in `src/systems/<id>/` and `src/systems/<id>/index.ts` default-exports a `SystemModule`:

```ts
export interface SystemModule {
  id: string;            // folder name
  priority: number;      // lower runs first
  create(game: Game): GameSystem;
}
export interface GameSystem {
  init(): Promise<void> | void;
  fixedUpdate?(dt: number): void;          // 60 Hz, after physics inputs are applied, before world.step()
  update?(dt: number, alpha: number): void; // every render frame
  dispose?(): void;
  serialize?(): unknown;                    // save/load hook
  deserialize?(data: unknown): void;
  debugActions?: Record<string, (...args: unknown[]) => unknown>;
}
```

- `Game` discovers modules with `import.meta.glob('../systems/*/index.ts')` (lazy), filters by the `?systems=` URL parameter (default: all), sorts by priority, and awaits `init()` in order.
- Every call into a system is wrapped in try/catch. On throw: `console.error('[system:<id>]', err)`, and that system is **disabled** for the rest of the session. The game keeps running.
- Priority bands: core/input/time 0–49, physics 50, world 100, player 150, camera 190, vehicles 200, traffic 210, peds 220, pickups 230, weapons 300, damage 310, particles 320, wanted 400, police 410, economy 500, shops 510, robbery 520, casino 530, interiors 540, missions 600, cutscene 610, companion 620, characters 630, navigation 700, ui 800, audio 900.

### 3.2 Game loop

Fixed-step physics at 60 Hz with an accumulator and a maximum of 3 substeps per frame; `update(dt, alpha)` runs once per render frame with interpolation alpha. `game.time.scale` implements slow-mo (affects fixed steps and dt, not UI). Delta clamped to 100 ms.

### 3.3 Events

`game.events` is a typed emitter over the `GameEvents` interface in `src/core/Events.ts`. Systems add events **without editing that file** via declaration merging in their own `events.ts`:

```ts
declare module '../../core/Events' {
  interface GameEvents {
    'vehicle:entered': { vehicle: Vehicle; ped: Ped | Player };
  }
}
```

Naming: `domain:pastTenseVerb`. Core events defined by Agent 1 (binding names and payloads):

- `crime:committed` `{ type: CrimeType; position: Vector3Like; actor: 'player' | 'companion'; loud: boolean }` where `CrimeType = 'shot_fired' | 'kill_ped' | 'kill_cop' | 'carjack' | 'steal_police_vehicle' | 'rob_ped' | 'rob_store' | 'hit_cop_vehicle' | 'hit_cop' | 'explosion' | 'attack_helicopter' | 'kill_swat'`
- `player:died` `{ position }`, `player:respawned` `{ position; reason: 'wasted' | 'busted' }`, `player:damaged` `{ amount; source }`
- `time:hourChanged` `{ hour }`, `time:dayChanged` `{ day }`
- `interior:entered` `{ id }`, `interior:exited` `{ id }`
- `economy:cashChanged` `{ cash; delta; reason }`
- `notification` `{ text; kind?: 'info' | 'money' | 'warning' }`

### 3.4 Shared read model (`game.state`)

UI reads, systems write. Agent 1 defines this exactly in code; fields at minimum:

```ts
interface GameState {
  cash: number; cashDelta: number;                 // economy
  health: number; armor: number;                   // player
  wanted: { level: number; state: 'none'|'responding'|'searching'|'active'; searchCenter: V3|null; searchRadius: number; hotScenes: {position: V3; radius: number}[] };
  weapon: { id: string; ammoInMag: number; ammoReserve: number; isReloading: boolean } | null;
  vehicle: { id: string; speedKmh: number; health: number } | null;
  objective: string | null;                        // missions
  prompts: { key: string; text: string }[];        // interaction prompts
  timeOfDay: number; day: number;                  // 0..24
  playerPos: V3; playerHeading: number;            // radians
  blips: { id: string; kind: BlipKind; position: V3; label?: string }[];
  waypoint: V3 | null; route: V3[];                // navigation
  interior: string | null; chips: number;          // casino
  activeCharacter: 'lucia' | 'jason'; abilityCharge: number;
  stats: Record<string, number>;
}
```

### 3.5 Entities and physics

- `game.physics` wraps Rapier: `world`, `register(collider, entity)`, `entityOf(colliderHandle)`, `raycast(origin, dir, maxToi, opts)` returning `{ entity, point, normal, toi, collider }`, collision groups enum (`Groups.STATIC | PLAYER | VEHICLE | PED | PICKUP | TRIGGER | BULLET_BLOCKER`), and `onCollision(handleA, handleB, started)` fan-out via events `physics:contactStarted` / `physics:contactEnded` with resolved entities.
- Interfaces in `src/core/entities.ts`: `Entity { id; kind; object3d; position; isAlive }`, `IDamageable { applyDamage(info: DamageInfo): void }` where `DamageInfo = { amount; type: 'bullet'|'melee'|'explosion'|'vehicle'|'fall'|'fire'; point?; direction?; part?: 'head'|'body'|'limb'; source?: Entity|'player' }`, `IEnterable { canEnter(who); enter(who); exit(who) }`.
- Systems own pools of their entities; every collider is registered so raycasts and contacts resolve to entities.

### 3.6 Plugin points other agents rely on

- **Vehicle drivers:** `Vehicle.driver: VehicleDriver | null` with `VehicleDriver { update(vehicle: Vehicle, dt: number): void; dispose?(): void }`. Agent 2 provides `TrafficDriver`; Agent 8 provides `PoliceDriver`; Agent 10 may provide scripted drivers.
- **Ped brains:** `Ped.brain: PedBrain` with `PedBrain { update(ped: Ped, ctx: PedContext, dt: number): void; onEvent?(ped, event): void }`. Agent 3 provides `CivilianBrain`, `GangBrain`; Agent 8 provides `CopBrain`, `SwatBrain`; Agent 9 provides `ClerkBrain`, `GuardBrain`; Agent 10 provides `CompanionBrain`.
- **NPC firing:** Agent 4 exposes `WeaponSystem.npcFire(shooter: Entity, weaponId: string, origin: V3, direction: V3, accuracy: number)` so cops, gangs, guards, and the companion can shoot.
- **Aim target:** Agent 4 emits `weapon:aimedAt` `{ target: Entity; seconds: number }` every frame the player aims at an entity (cower/rob/clerk logic reacts).
- **Interiors:** Agent 1 provides the framework `game.interiors.register(def)`, `enter(id)`, `exit()`, door markers generated from `data/landmarks.ts`; `InteriorDefinition { id; build(scene, physics): InteriorHandle; spawnPoint; exitPoint }`. Agents 6 and 9 register interiors.
- **Economy:** Agent 1 ships a minimal wallet `game.economy` with `cash`, `add(amount, reason)`, `trySpend(amount, reason): boolean`; Agent 9 owns and extends it afterwards without breaking the API.
- **Camera:** Agent 1 provides `game.camera` with modes `thirdPerson | aim | vehicle | cinematic | map`, `setTarget(target: CameraTarget)` where `CameraTarget { position: V3; quaternion: Quaternion; velocity: V3; kind: 'onFoot'|'vehicle' }`, `setMode(mode)`, `shake(strength, seconds)`, `fovKick(delta)`, and a cinematic API `cinematic.moveTo(pos, lookAt, seconds)` / `cinematic.orbit(...)`.
- **Player mount/dismount:** `game.player.mount(vehicle: IEnterable & Entity, seat)` hides the character controller and parents the player; `game.player.dismount(position)`. Agent 2 calls these.
- **Debug actions and autotests:** `window.__game` is the `Game`; `game.debug.run('systemId.action', ...args)`; URL `&actions=a,b` runs actions after ready; URL `&autotest=<name>` imports `src/debug/autotests/<name>.ts` and calls its default export `(game) => Promise<void>`. The harness sets `window.__GAME_READY__ = true` after all systems init.
- **Scripted input:** `game.input.press(code, ms)`, `hold(code)`, `release(code)`, `look(dx, dy)`, `setAim(bool)`, `setFire(bool)`, `setMouseWorld(v3)`; autotests drive the game only through these.
- **Audio:** `game.audio.play(id: SoundId, opts?: { position?: V3; volume?: number; pitch?: number })`, `loop(id, opts) => LoopHandle`, `stop(handle)`, `setLoopParams(handle, { pitch?, volume?, position? })`, `engine(handle, rpm01, throttle01, speedMs)`, `siren(handle, on)`, `setListener(pos, quat, vel)`, `setBusVolume(bus, v)`. `SoundId` is the union in `src/systems/audio/sounds.ts` (Agent 1 writes the enumerated list from SYSTEM_PROMPT §16; Agent 7 implements every id). Unknown ids no-op with a one-time warning.

### 3.7 Isolation testing

`?systems=world,player,camera,vehicles` loads only the named systems plus core. Every agent's system must handle missing peers gracefully (`game.systems.get('peds')` may be undefined). `npm run smoke -- --systems=<ids> --name=<label> [--autotest=<name>] [--actions=a,b] [--seconds=10]` starts Vite on a random port, opens headless Chromium (new headless with GPU flags; if WebGL is unavailable it exits with a clear message), waits for `__GAME_READY__`, runs, records console errors/warnings and page errors, saves `docs/screenshots/<label>.png` and `docs/screenshots/<label>.log`, and exits non-zero on any error or `AUTOTEST FAIL`. Headless frame rates are not representative; performance claims must come from a real browser or from `game.perf` counters (draw calls, triangles, bodies, entities) which are printed in the log.

---

## 4. Ownership matrix

| Agent | Owned paths (create/edit) |
|---|---|
| 1 Foundation | `package.json`, root configs, `index.html`, `src/main.ts`, `src/core/**`, `src/physics/**`, `src/world/**` (except `interiors/<content>`), `src/world/interiors/Interiors.ts`, `src/systems/world/**`, `src/systems/player/**`, `src/systems/camera/**`, `src/systems/economy/**` (until Stage 3), `src/systems/audio/index.ts` + `sounds.ts` (stub only, until Agent 7), `src/entities/Player.ts`, `src/gfx/Textures.ts`, `src/gfx/Materials.ts`, `src/gfx/BuildingBuilder.ts`, `src/gfx/PedBuilder.ts` (v1 only, then Agent 3), `src/data/districts.ts`, `src/data/landmarks.ts`, `src/debug/**` (harness; autotests folder shared), `scripts/**`, `src/ui/styles.css` (base variables only), `docs/CONTRACTS.md`, `docs/ARCHITECTURE.md`, `docs/PROGRESS.md`, `README.md`, `CONTROLS.md` |
| 2 Vehicles & Traffic | `src/systems/vehicles/**`, `src/systems/traffic/**`, `src/entities/Vehicle.ts`, `src/gfx/CarBuilder.ts`, `src/data/vehicles.ts`, `src/debug/autotests/vehicles*.ts`, `src/debug/autotests/traffic*.ts` |
| 3 Pedestrians | `src/systems/peds/**`, `src/systems/pickups/**`, `src/entities/Ped.ts`, `src/entities/Pickup.ts`, `src/gfx/PedBuilder.ts` (from Stage 2), `src/data/peds.ts`, `src/debug/autotests/peds*.ts` |
| 4 Weapons, Damage & FX | `src/systems/weapons/**`, `src/systems/damage/**`, `src/systems/particles/**`, `src/gfx/Effects.ts`, `src/data/weapons.ts`, `src/ui/WeaponWheel.ts` + `src/ui/weapon-wheel.css`, `src/debug/autotests/weapons*.ts` |
| 5 UI | `src/systems/ui/**`, `src/systems/navigation/**`, `src/ui/**` (except WeaponWheel, casino UI, phone), `src/debug/autotests/ui*.ts` |
| 6 Casino | `src/casino/**`, `src/systems/casino/**`, `src/world/interiors/CasinoInterior.ts`, `src/ui/casino/**`, `tests/casino/**`, `src/debug/autotests/casino*.ts` |
| 7 Audio | `src/systems/audio/**` (replaces the stub, keeps the API), `tests/audio/**`, `src/debug/autotests/audio*.ts` |
| 8 Wanted & Police | `src/systems/wanted/**`, `src/systems/police/**`, `src/data/police.ts`, `src/debug/autotests/wanted*.ts`, `src/debug/autotests/police*.ts` |
| 9 Economy, Shops, Robberies, Phone | `src/systems/economy/**`, `src/systems/shops/**`, `src/systems/robbery/**`, `src/systems/interiors/**`, `src/world/interiors/**` (except Casino), `src/data/prices.ts`, `src/data/shops.ts`, `src/ui/phone/**`, `src/debug/autotests/shops*.ts`, `src/debug/autotests/robbery*.ts` |
| 10 Missions & Story | `src/missions/**`, `src/systems/missions/**`, `src/systems/cutscene/**`, `src/systems/companion/**`, `src/systems/characters/**`, `src/data/dialogue.ts`, `src/data/story.ts`, `src/debug/autotests/missions*.ts` |

`docs/progress/<agentid>.md` is owned by each agent (`foundation`, `vehicles`, `peds`, `weapons`, `ui`, `casino`, `audio`, `wanted`, `shops`, `missions`).

---

## 5. Agent briefs

Each brief lists scope, deliverables, public API, and acceptance criteria. The spec details (numbers, behaviors) are in the referenced `SYSTEM_PROMPT.md` sections; follow them exactly.

### Agent 1 — Foundation / Architect (Stage 1)

Scope: `SYSTEM_PROMPT.md` §1, §2, §3, §4 (player and camera), Phase 0 and Phase 1 acceptance, plus everything in §3 of this file.

Deliverables:
- Vite + TypeScript strict project with **all** dependencies installed: `three`, `@types/three`, `@dimforge/rapier3d-compat`, `vitest`, `playwright` (+ `npx playwright install chromium`). Scripts: `dev`, `build`, `preview`, `typecheck`, `test`, `smoke`.
- `src/core`: `Game` (loop, discovery, isolation, try/catch), `Events`, `Input` (pointer lock + scripted), `Time` (clock, day/night time, slow-mo scale), `Config`, `Pool`, `SpatialHash`, `Random` (seeded), `Save` (registry over `serialize/deserialize`, localStorage, versioned), `entities.ts` interfaces, `GameState`.
- `src/physics/Physics.ts` wrapper per §3.5.
- World: `RoadGraph` (nodes, edges, lanes, sidewalks, crosswalks, parking spots, `astar`, nearest queries, random points), `CityGenerator`, `Districts`, `Landmarks` (all landmarks from spec with door markers), `BuildingBuilder` (merged chunks, canvas window atlas with night emissive), `Props`, `Water` (with trigger volumes and a `game.world.isInWater(pos)` query), bridges, `DayNight` (24-min day, sky, sun/moon, fog, lamps/neon at dusk), `Interiors` framework.
- Player (`entities/Player.ts` + `systems/player`): kinematic character controller, movement speeds, jump, aim-mode strafing, procedural walk/run/aim animation using `PedBuilder` v1, health/armor with `IDamageable`, regen, death → 4 s → respawn at nearest hospital with 10% cash loss (`player:died`, `player:respawned`), `mount/dismount`. Swimming can be deferred to Agent 1's final pass.
- Camera system per §4 with all modes and the cinematic API.
- Minimal wallet `game.economy`.
- Audio API stub (`src/systems/audio/index.ts`, `sounds.ts` with the complete `SoundId` union covering every sound named in `SYSTEM_PROMPT.md` §16 and the needs of §5–§15).
- Debug: F3 perf overlay (fps, ms, draw calls, triangles, bodies, entity counts via `game.perf`), `window.__game`, debug actions, `&actions=`, `&autotest=` harness, `__GAME_READY__`.
- `scripts/smoke.mjs` per §3.7, plus `src/debug/autotests/foundation.ts` covering: game ready with zero errors, player walks 20 m forward via scripted input, jump changes height, A* between 20 random node pairs succeeds, every landmark door is within 15 m of a road node, static draw calls < 400, time advances and lamps turn on at dusk.
- `docs/CONTRACTS.md` (exact code-level contracts), `docs/ARCHITECTURE.md` (folder layout, loop, how to add a system, how to test in isolation, how debug/autotest works), `docs/progress/foundation.md`, `CONTROLS.md`, `README.md` (run instructions).
- `git init`, `.gitignore`, one commit per phase (`feat(phase-0): ...`, `feat(phase-1): ...`).

Acceptance (Phase 0 + Phase 1 from `SYSTEM_PROMPT.md` §19), verified by the smoke run and by viewing `docs/screenshots/foundation-*.png`: sunset over Vice Beach with art-deco pastels and neon coming on, downtown skyline, causeway bridge, swamp edge, player visible in third person.

### Agent 2 — Vehicles & Traffic (Stage 2)

Scope: `SYSTEM_PROMPT.md` §5 and §6. Isolation set: `world,player,camera,vehicles,traffic`.

Deliverables: `Vehicle` entity (`IDamageable`, `IEnterable`, `driver` plugin), `CarBuilder` (all archetypes in §5 with per-instance color, light bar for police with animated emissive, headlights/taillights), `data/vehicles.ts` with tuned handling per archetype, `VehicleSystem` (raycast vehicle per §5, enter/exit incl. carjack hook that asks `peds` (if present) to eject the driver as a fleeing ped, damage tiers, smoke/fire/explosion with radius damage through `DamageSystem` if present, tire shots, horn/lights/camera toggle, camera target updates, `game.audio` engine/skid/impact/explosion calls), parked cars from `roadGraph` parking spots by district, `TrafficSystem` (kinematic lane followers, lights, car-following, honking, rerouting, react to gunfire by stopping/fleeing, convert to dynamic on contact, instanced rendering, pooling, spawn/despawn rules).

Public API: `VehicleSystem.spawn(archetypeId, position, heading, opts?: { color?, driver?, parked? }): Vehicle`, `nearestEnterable(pos, radius)`, `Vehicle.enter/exit`, `Vehicle.setDriver(driver)`, `Vehicle.controls { throttle, brake, steer, handbrake }` (set by player input or a driver), `Vehicle.state { speedMs, rpm01, health, onFire, exploded, occupants }`, events `vehicle:entered`, `vehicle:exited`, `vehicle:damaged`, `vehicle:exploded`, `vehicle:collided`, `vehicle:carjacked`; `TrafficSystem.setDensity(n)`, `TrafficSystem.panicNear(pos, radius)`.

Acceptance: Phase 2 and the traffic half of Phase 3 in `SYSTEM_PROMPT.md` §19. Autotests: spawn sedan, enter, drive 100 m along a lane at > 12 m/s without leaving the road, handbrake turn yields yaw rate, collide with a wall at 25 m/s → health drops, explode on demand → radius damage, traffic count stabilizes at target density with zero stuck-for-60 s cars, traffic stops on red at 3 sampled intersections.

### Agent 3 — Pedestrians (Stage 2)

Scope: `SYSTEM_PROMPT.md` §7 plus ped robbing and cash pickups. Isolation set: `world,player,camera,peds,pickups`.

Deliverables: `Ped` entity (`IDamageable`, brain plugin, state machine per §7), `PedBuilder` v2 (owns it from now; keep the API Player uses), six-part instanced rendering with per-instance colors and distance-gated animation, kinematic-alive / dynamic-ragdoll swap, knockdown/get-up, `data/peds.ts` types and palettes with district and time-of-day density, `CivilianBrain`, `GangBrain` (Docks zones; returns fire via `weapons.npcFire` if present), `IdleGroup` behavior, `COWER` on `weapon:aimedAt` → cash pickup after 1.5 s → `crime:committed rob_ped`, `FLEE` from `crime:committed` (loud within 40 m) and explosions, `CALL_POLICE` (10% of witnesses; emits `ped:reportedCrime` after 4 s for Agent 8), `FIGHT` when attacked, ragdoll on vehicle contact via `physics:contactStarted`, spawn/despawn on sidewalks out of view, spatial hash usage, `PickupSystem` (cash/weapon/armor/health pickups with bob/spin, magnet radius, `pickup:collected`).

Public API: `PedSystem.spawn(typeId, position, opts?: { brain?, heading?, weaponId? }): Ped`, `PedSystem.query(pos, radius, filter?)`, `PedSystem.witnessesNear(pos, radius): Ped[]`, `Ped.setBrain`, `Ped.state`, `Ped.type`, `Ped.enterVehicle(vehicle, seat)` / `exitVehicle()` (drives `IEnterable` if vehicles present), events `ped:died`, `ped:knockedDown`, `ped:cowered`, `ped:fled`, `ped:reportedCrime`, `ped:attacked`; `PickupSystem.spawn(kind, position, value)`.

Acceptance: ped half of Phase 3 plus the robbing part of Phase 4 in `SYSTEM_PROMPT.md` §19. Autotests: 80 peds near player with zero errors; aim at a ped (simulated `weapon:aimedAt`) → cower → pickup spawns → walking into it adds cash; a fake `crime:committed shot_fired` makes peds within 40 m flee and at least one call police; applying vehicle-type damage ragdolls a ped and it despawns within 25 s; peds stay on sidewalks (sample 50 positions, ≥ 90% within 3 m of a sidewalk polyline).

### Agent 4 — Weapons, Damage & FX (Stage 2)

Scope: `SYSTEM_PROMPT.md` §8 plus the damage and particle systems. Isolation set: `world,player,camera,weapons,damage,particles` (spawn debug target dummies implementing `IDamageable` for testing; if `peds` is present, use real peds).

Deliverables: `data/weapons.ts` exactly per the table, `WeaponSystem` (inventory/ammo, equip, hitscan with spread through `physics.raycast`, fire rates, reload, recoil, camera aim mode toggling via `game.camera.setMode`, sniper scope FOV, soft aim-assist toggle, firing from vehicle window when `game.state.vehicle`, melee combo, `npcFire`, `weapon:aimedAt`, `weapon:fired`, `crime:committed shot_fired` with `loud`), `DamageSystem` (single entry point `apply(target, info)`, headshot ×3, armor absorption for player, kill attribution → `crime:committed kill_ped|kill_cop|kill_swat` by victim type, explosion radius damage helper `explode(position, radius, damage, impulse)`), `ParticleSystem` (pooled GPU-friendly sprites: muzzle flash, tracer lines, impact sparks/dust/blood puffs by surface, smoke, fire, explosion fireball, shell casings optional) with `game.audio` calls, `WeaponWheel` UI (hold Tab, slow-mo via `game.time.scale`, radial layout, number keys, scroll), crosshair + hit markers (white hit / red kill) in `src/ui/weapon-wheel.css` or a small crosshair element you own.

Public API: `WeaponSystem.give(weaponId, ammo)`, `equip(weaponId)`, `addAmmo`, `current`, `npcFire(...)`, `DamageSystem.apply(...)`, `DamageSystem.explode(...)`, `ParticleSystem.emit(effectId, position, opts)`, events `weapon:fired`, `weapon:reloaded`, `weapon:equipped`, `weapon:aimedAt`, `damage:applied`, `entity:killed`.

Acceptance: Phase 4 combat items in `SYSTEM_PROMPT.md` §19 (wallet/HUD excluded). Autotests: give pistol; fire at a dummy 20 m away with scripted aim → damage applied; headshot dummy takes 3×; reload timing within ±10% of spec; SMG fires ~13 rounds/s; shotgun pellets spread; `explode` damages dummies within radius and not beyond; weapon wheel opens and equips via scripted input; `crime:committed shot_fired` emitted per shot.

### Agent 5 — UI (Stage 2)

Scope: `SYSTEM_PROMPT.md` §12 and §13 (all HUD, minimap, full map, GPS, notifications, pause menu, title screen, WASTED/BUSTED/MISSION PASSED/FAILED cards, dialogue/subtitle component) plus the `navigation` system (waypoint + A* route). Isolation set: `world,player,camera,navigation,ui`.

Deliverables: `systems/ui` (mounts overlays, reads `game.state` each frame, never writes game state except UI-owned fields like `waypoint` via navigation), `Minimap` (2D canvas from `roadGraph` and district data, rotating with heading, zoom by speed, all icon kinds, search circle, hot scenes, route), `FullMap` (pan/zoom, legend, click to set waypoint, Esc/M), `navigation` (waypoint, A* route via `roadGraph.astar`, recompute on stray, distance text, mission blips API), `HUD` (cash with tick animation and deltas, stars with the three color states and flash, weapon/ammo, health/armor bars under minimap, objective text with yellow nouns, prompts, robbery progress bar hook), `Notifications`, `PauseMenu` (resume/map/stats/settings persisted to localStorage/save/load via `game.save`/controls/quit), `TitleScreen` (LEONIDA, cinematic camera over Vice Beach via `game.camera.cinematic`, New Game/Continue), `Cards` (`ui.card(kind, text, seconds)` with letterbox + slow-mo via `game.time.scale` + desaturate CSS filter), `Dialogue` (`ui.dialogue.show(lines: {speaker, text, seconds}[])`, skip with Space), `styles.css` additions in your own CSS files (import them from your components; do not edit the base file).

Public API: `game.ui.notify(text, kind)`, `game.ui.card(...)`, `game.ui.dialogue.show/hide`, `game.ui.prompt(key, text)/clearPrompt(key)`, `game.ui.setObjective(text|null)`, `game.ui.progressBar.show(label, fraction)/hide()`, `game.ui.pause()/resume()`, `game.navigation.setWaypoint(v3|null)`, `addBlip(blip)/removeBlip(id)`, `routeTo(v3): V3[]`.

Acceptance: Phase 6 in `SYSTEM_PROMPT.md` §19 plus cards and dialogue. Autotests: set a waypoint 600 m away → route has > 5 points and ends within 15 m; minimap canvas renders non-empty pixels; mutate `game.state.wanted` to each state and verify star DOM classes; `notify` shows and expires; card shows and restores time scale; screenshot review of HUD layout at 1440×900 and 1920×1080.

### Agent 6 — Casino (Stage 2)

Scope: `SYSTEM_PROMPT.md` §11 in full. Isolation set: `world,player,camera,economy,interiors,casino,ui` (if `ui` is absent, your own casino UI must still function).

Deliverables: pure-logic modules with vitest coverage (`Blackjack.ts` with shoe, all actions, dealer rules, payouts, edge cases: split aces one card, double only on first two, push, natural vs 21 after split; `Roulette.ts` American wheel and every bet type and payout; `Slots.ts` weighted reels and paytable with an RNG injection for deterministic tests; `LuckyWheel.ts` prize table and daily lock keyed to `game.time.day`), `CasinoInterior` registered via `game.interiors` (lobby, cashier cage, two blackjack tables with limits, roulette table, slot bank, wheel, bar, guard peds if `peds` present), 3D props from primitives with canvas-textured cards/chips/wheel, `CasinoUI` (HTML controls: cashier buy/cash-out, bet chips, blackjack action bar, roulette betting layout, slot lever, wheel spin; chip balance display; results text), animations (card deal, wheel + ball spin ~6 s, reel stagger), `game.audio` calls, guard reaction to drawn weapon (emits `crime:committed` hooks and locks doors for a day), Lucky Wheel vehicle prize spawns a sports car outside via `vehicles` if present.

Public API: `game.casino.enter()`, `game.casino.play(game: 'blackjack'|'roulette'|'slots'|'wheel')`, `game.casino.chips`, events `casino:won`, `casino:lost`, `casino:entered`, `casino:exited`.

Acceptance: Phase 8 in `SYSTEM_PROMPT.md` §19. `npm test` covers the four pure modules with ≥ 40 assertions total including payout math. Autotests: enter interior → buy 1,000 chips → cash drops by 1,000 → play a scripted blackjack hand with a rigged shoe → chip balance changes by the exact expected amount; roulette straight bet on a rigged spin pays 35:1; slots rigged jackpot pays 500×; wheel spins once, second spin same day refused.

### Agent 7 — Audio (Stage 2)

Scope: `SYSTEM_PROMPT.md` §16 in full, implementing every `SoundId` in `src/systems/audio/sounds.ts` behind the API in §3.6. Isolation set: `world,player,camera,audio`.

Deliverables: Web Audio graph (master, buses sfx/engine/ambient/ui/music with gains persisted in settings), positional audio via `PannerNode` with distance rolloff and listener updates, synthesized sounds: every weapon fire and reload, impacts by material, footsteps by surface and speed, engine model (two detuned saws + noise + lowpass + rpm mapping + gear-shift feel), tire screech, horn, siren wail with LFO and doppler-ish pitch, explosion, ped yelps, casino sounds (chip clink, card slide, reel tick, jackpot, wheel tick), UI blips, wanted-lost jingle, mission-passed sting, district ambient beds (waves, city hum, swamp insects) with crossfade by district and time of day, one generated synthwave radio station with a small sequencer (on/off via `game.audio.radio(on)`), voice pool with limits and priority, autoplay policy handling (resume on first input), graceful no-op when `AudioContext` is unavailable (headless).

Acceptance: `npm test` covers the engine rpm mapping and voice pool limits; autotest plays every `SoundId` once with zero errors and confirms the graph node counts stay bounded after 500 plays (pool works); listening review notes in your progress doc for each family with the parameter choices.

### Agent 8 — Wanted & Police (Stage 3)

Scope: `SYSTEM_PROMPT.md` §9 in full. Uses vehicles, peds, weapons, ui, audio, navigation.

Deliverables: `WantedSystem` (level, state machine with the three star states, crime table, witness logic using `peds.witnessesNear` and cop line of sight, `ped:reportedCrime` handling, search circle, hot scenes with 120 s expiry, evasion timers, heat/criminal profile, respray hook `wanted.clear(reason)`, `game.state.wanted` writes, `wanted:changed` events, audio cues), `PoliceSystem` (unit pool with dispatch by level, `PoliceDriver` for cruisers/SWAT vans using `roadGraph.astar` re-path every 2 s and ramming, `CopBrain`/`SwatBrain` for foot cops with approach/arrest/shoot/cover behaviors and `weapons.npcFire`, arrest detection → BUSTED via `ui.card` and `player` respawn at police station with 10% cash loss, helicopter kinematic entity with spotlight and gunner, roadblocks with spike strips, level caps, cop LOS raycasts throttled), `data/police.ts`, respray shop interior hook is Agent 9's but `wanted.clear` is yours.

Acceptance: Phase 5 in `SYSTEM_PROMPT.md` §19. Autotests: `crime:committed shot_fired` with a witness → level 1 in state `responding`, becomes `active` when a cop gets LOS; kill a cop → +2; reach 4 via debug → SWAT van and helicopter exist; teleport player 400 m away and out of LOS → state `searching` then cleared after the timer; stand still at level 1 next to a cop → BUSTED card and respawn at a police station; hot scene re-entry re-raises.

### Agent 9 — Economy, Shops, Robberies, Chop Shop, Phone (Stage 3)

Scope: `SYSTEM_PROMPT.md` §10 (all), the respray shop from §9, phone from §13, and the interiors content for Bullet Barn, 24/7 ×4, Gas Station, Waffle Hut, Safehouse, Respray, Chop Shop marker, Jewelry Store and Bank shells (guards, for Agent 10). Uses everything from Stage 2.

Deliverables: `EconomySystem` (owns wallet from now; cash tick, deltas, reasons, death/arrest fee hooks, stats), `data/prices.ts`, `data/shops.ts`, `ShopSystem` (Bullet Barn UI with buy/sell weapons/ammo/armor via `weapons.give`; food; clerk pulls shotgun on drawn weapon), `RobberySystem` (aim-at-clerk detection via `weapon:aimedAt`, hands up, bag-filling progress via `ui.progressBar`, gunshot speed-up, 25% shotgun clerk, register cash pickup, `crime:committed rob_store` on exit, closed-for-a-day state), `ClerkBrain`, `GuardBrain`, `ChopShop` (marker, valuation table, daily wanted vehicle, vehicle despawn and payout), respray (`wanted.clear` when not seen, cost, recolor), `Phone` (Up arrow overlay: Contacts list fed by `game.missions` if present, Map shortcut, Snapmatic canvas capture to PNG download, parody feed generated from `crime:committed`, `casino:won`, `economy:cashChanged`), all interiors registered through `game.interiors` with door markers and blips through `navigation.addBlip`.

Acceptance: Phase 7 in `SYSTEM_PROMPT.md` §19. Autotests: buy AR → cash drops and inventory has it; rob a 24/7 with scripted aim → progress completes → pickup → exit → `wanted.level === 2`; sell a spawned sports car at 100% health → +$4,000; respray with wanted 2 and no cop LOS → level 0 and cash −$2,000; phone opens and Snapmatic produces a data URL.

### Agent 10 — Missions, Story, Cutscenes, Companion, Characters (Stage 4)

Scope: `SYSTEM_PROMPT.md` §14 and §15 in full.

Deliverables: `MissionSystem` (objective primitives listed in §14 with fail conditions, contact markers and phone start, rewards, autosave via `game.save`, retry flow, `game.state.objective`, mission blips), `CutsceneSystem` (letterbox via `ui.card` or your own overlay, `camera.cinematic` moves, `ui.dialogue` lines from `data/dialogue.ts`, Space to skip), all eight missions as separate files with the exact beats in §14, `CompanionBrain` (Jason/Lucia follow, enter passenger seat, `weapons.npcFire` at hostiles and pursuing cops, downed/revive), `characters` system (Q character wheel with slow-mo, sky-up/sky-down transition, separate state per character, palette/stat differences, Caps Lock adrenaline ability with recharge and the two variants), `data/story.ts`, post-finale repeatable jobs (chop shop wanted vehicle and store robberies are Agent 9's; you wire them into contacts).

Acceptance: Phase 9 in `SYSTEM_PROMPT.md` §19. Autotests: start mission 1 via debug, complete each objective via scripted input/teleport helpers, MISSION PASSED card and reward; fail a mission by killing the player and confirm retry; run missions 2–8 in a scripted fast mode (`missions.debugFastForward`) to completion with zero errors; character switch changes `game.state.activeCharacter` and camera transition completes; ability sets `time.scale` to 0.3 and restores.

### Agent 1 (resumed) — Integration pass 1

Enable all Stage 2 systems together (`npm run smoke` with no `?systems=`), apply every **Integration request** from `docs/progress/*.md`, fix cross-system bugs (typical: event payload mismatches, entity registration, priority ordering, missing peers), get `npm run typecheck`, `npm run test`, and every agent's autotest green in the combined build, verify screenshots, write `docs/PROGRESS.md` (compiled from progress docs), commit `feat(stage-2): integrate vehicles, traffic, peds, weapons, ui, casino, audio`.

### Agent 1 (resumed) — Integration pass 2 / Polish / QA

Apply Stage 3–4 integration requests, complete `Save/Load` end to end (every system's `serialize/deserialize`, Continue on title, autosave), settings menu wiring, swimming, weather (optional), performance pass against `SYSTEM_PROMPT.md` §20 with real-browser numbers documented, juice pass, `README.md` with a screenshots section, all autotests green in the full build, `npm run build` succeeds and `npm run preview` loads, final commit. Confirm the Definition of Done in `SYSTEM_PROMPT.md` §24 item by item in `docs/PROGRESS.md`.
