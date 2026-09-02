# LEONIDA — Progress

LEONIDA is a GTA VI-style open-world crime sandbox in the browser: a procedurally generated Vice City (six districts, causeway, water, 24-minute day/night) built with Three.js + Rapier (compat), Vite and strict TypeScript, with driving, traffic, pedestrians, hitscan combat, a six-star wanted system, shops/robberies/chop shop, the Leonida Palms casino, and an eight-mission story for Lucia and Jason. Run it with `npm install && npm run dev`, open the printed URL, click the canvas to lock the mouse; `Esc` pauses. Controls: [CONTROLS.md](../CONTROLS.md). **Verification status of this build:** `npm run typecheck` (0 errors project-wide), `npm run test` (vitest, 16 files / 127 tests at the Stage 2 integration pass, plus the Stage 3/4 suites `tests/shops`, `tests/wanted`, `tests/missions`) and a read-only code review ([integration-stage2.md](progress/integration-stage2.md)) have passed. **No system has been run in a browser this session** (browser automation was unavailable to every agent); every autotest below is written but unexecuted. Browser QA is the next step, and first-run issues are expected. To report one: open DevTools, copy the console errors, run `__game.debug.run('core.errors')` (the harness's collected error list), `__game.debug.run('core.systems')` (which systems got disabled after a throw), and `__game.debug.run('core.perf')`, and note the URL parameters you used.

This document is compiled from `docs/progress/*.md` (one per agent). Where they disagree with the code, the code wins; report it.

---

## 1. System status

Status legend: **Done-untested** = implemented, typechecked, unit-tested where applicable, has an autotest, not yet run in a browser. **In progress** = no dedicated progress doc (`docs/progress/foundation.md` was never written; the foundation systems are covered only by `src/debug/autotests/foundation.ts` and the Stage 2 integration review). **Missing** = not implemented.

| System id | Status | Owner doc | Key debug actions | Autotest(s) | Top known issues |
|---|---|---|---|---|---|
| `world` | In progress (no progress doc; foundation autotest + integration review only) | [integration-stage2.md](progress/integration-stage2.md) | `world.goto:<landmarkId>`, `world.landmarks`, `world.stats`, `world.night`, `world.inWater`, `world.ground` | `foundation` (`astar-20-random-pairs`, `landmark-doors-near-road-nodes`, `static-draw-calls-under-400`, `dusk-turns-lamps-on`, `water-query`) | Async city build can finish after dependent systems init (peds allow 8 s). Draw-call budget unmeasured. |
| `player` | In progress (no progress doc) | [integration-stage2.md](progress/integration-stage2.md) | `player.teleport:<x>:<z>[:<heading>]`, `player.pos`, `player.damage:<n>`, `player.kill[:busted]`, `player.heal`, `player.armor:<n>`, `player.face:<rad>` | `foundation` (`walk-20m`, `jump-raises-player`, `damage-armor-death-respawn`) | Hospital fee is 10 % (player) vs 5 % min $100 (economy design); only one is charged. `Continue` from an airborne save falls one frame (no ground snap in `deserialize`). |
| `camera` | In progress (no progress doc) | [integration-stage2.md](progress/integration-stage2.md) | `camera.setOrbit`, `camera.setMode:<mode>`, `camera.shake`, `camera.moveTo`, `camera.orbit`, `camera.restore`, `camera.state` | `foundation` (`camera-modes-and-cinematic`) | Sniper `fovKick(-25)` fights the aim FOV; snaps back over the camera's own recovery. `C` in vehicles is a no-op (camera owns distance). |
| `vehicles` | Done-untested | [vehicles.md](progress/vehicles.md) | `vehicles.spawn:<arch>:<x>:<z>[:<heading>]`, `vehicles.spawnHere:<arch>`, `vehicles.enterNearest`, `vehicles.exit`, `vehicles.explodeCurrent`, `vehicles.count`, `vehicles.state`, `vehicles.despawnAll` | `vehicles` | Single ground probe: nose over a kerb → brief loss of steering. Hitting a sleeping parked car damages the mover more until the parked body wakes. |
| `traffic` | Done-untested | [vehicles.md](progress/vehicles.md) | `traffic.setDensity:<n>`, `traffic.count`, `traffic.panic`, `traffic.lights`, `traffic.stuck`, `traffic.stalled[:<s>]` | `traffic` | Nose-to-tail queues restart with a lag chain and the tail catches the next red. No yielding at unlit intersections, no overtaking. |
| `peds` | Done-untested | [peds.md](progress/peds.md) | `peds.spawn:<type>:<x>:<z>`, `peds.spawnHere:<type>`, `peds.count`, `peds.stats`, `peds.state:<id>`, `peds.aimAt[:<id>]`, `peds.gunshot`, `peds.ragdollNearest`, `peds.killNearest`, `peds.ragdoll:<id>`, `peds.damage:<id>:<n>[:<type>]`, `peds.setDensity:<n>`, `peds.fastForward:<s>` | `peds` | Run-over kills bypass `damage.apply` → no `entity:killed`/`kill_ped` crime. Ragdolls jitter on moving vehicle roofs before freezing. |
| `pickups` | Done-untested | [peds.md](progress/peds.md) | `pickups.spawn:<kind>:<value>[:<weaponId>]`, `pickups.count`, `pickups.clear` | `peds` (cower → cash pickup → collect) | Pickups invisible inside interiors until the actors-root fix lands (still collectable). |
| `weapons` | Done-untested | [weapons.md](progress/weapons.md) | `weapons.give:<id>:<ammo>`, `weapons.equip:<id>`, `weapons.fire`, `weapons.reload`, `weapons.state`, `weapons.spawnDummy:<x>:<z>`, `weapons.dummies`, `weapons.clearDummies`, `weapons.aimAssist[:bool]` | `weapons` | No gun meshes / arm swing on the player rig. Drive-by fires along camera forward with no window-side restriction. |
| `damage` | Done-untested | [weapons.md](progress/weapons.md) | `damage.explode:<x>:<z>:<radius>:<damage>`, `damage.player` | `weapons` (explode falloff, headshot ×3) | Kill attribution only for hits routed through `damage.apply` (see peds). |
| `particles` | Done-untested | [weapons.md](progress/weapons.md) | `particles.emit:<id>:<x>:<z>`, `particles.count`, `particles.clear` | `weapons` (`particle pools bounded`) | Unsorted, unlit single-draw-call pools; no soft particles. |
| `navigation` | Done-untested | [ui.md](progress/ui.md) | `navigation.waypoint:<x>:<z>`, `navigation.clear`, `navigation.route`, `navigation.blips` | `ui` (`waypoint-route`, `landmark-blips`, `waypoint-clears-on-arrival`) | Blips not clustered on the full map; mission/contact labels only in the legend. |
| `ui` | Done-untested | [ui.md](progress/ui.md) | `ui.notify:<text>:<kind>`, `ui.card:<kind>:<text>[:<s>]`, `ui.dialogue`, `ui.objective:<text>`, `ui.progress:<f>`, `ui.title`, `ui.pause`, `ui.resume`, `ui.map`, `ui.state` | `ui` | Layout at 1440×900 / 1920×1080 never screenshot-reviewed. No HUD ability meter yet (missions request). |
| `audio` | Done-untested | [audio.md](progress/audio.md) | `audio.playAll`, `audio.voices`, `audio.radio:<bool>`, `audio.bus:<bus>:<vol>`, `audio.state` | `audio` | No listening pass; levels untuned. Loops started before the first gesture can bunch events at resume. |
| `casino` | Done-untested | [casino.md](progress/casino.md) | `casino.enter`, `casino.buy:<n>`, `casino.cashout:<n>`, `casino.chips`, `casino.play:<game>`, `casino.rig:<game>:<json>`, `casino.close`, `casino.lock[:days]`, `casino.unlock` | `casino` | Player/guards invisible inside until the actors-root fix. Banned kick-out shows one frame of the interior before fading. |
| `interiors` | Done-untested (framework: foundation; content: shops/casino) | [shops.md](progress/shops.md), [casino.md](progress/casino.md) | `interiors.enter:<id>`, `interiors.exit`, `interiors.list`, `interiors.current` | `foundation` (`interiors-enter-exit`), `shops` (`interiors-registered`) | **Player, peds and pickups are parented to `worldRoot`, which is hidden indoors → invisible in every interior** (shops uses `PedProxy` rigs as a stopgap). Casino station `E` and exit `E` both poll the same key. |
| `economy` | Done-untested | [shops.md](progress/shops.md) | `economy.add:<n>`, `economy.set:<n>`, `economy.get`, `economy.fee:<fraction>:<reason>`, `economy.stats` | `foundation` (`economy-wallet`), `shops` (`hospital-fee-once`) | Two hospital-fee owners (player 10 % vs economy 5 %); guarded against double charge, but the design number is not what is charged. |
| `shops` | Done-untested | [shops.md](progress/shops.md) | `shops.open:<gunstore\|phone>`, `shops.closeUi`, `shops.buy:<weaponId\|armor>`, `shops.food`, `shops.close:<storeId>`, `shops.reopen:<storeId>`, `shops.list`, `shops.status`, `shops.armClerk:<bool>`; also registers `phone.open/close/snap/app/feed`, `respray.spray/position` | `shops` | Clerk shotgun (accuracy 0.45 at close range) may be lethal fast. `data/weapons.ts` prices differ from `data/prices.ts` (store charges `prices.ts`). |
| `robbery` | Done-untested | [shops.md](progress/shops.md) | `robbery.start:<storeId>`, `robbery.status`, `robbery.skip:<s>`, `robbery.abort` | `robbery` | Register cash pickup invisible indoors (collectable). Store reopen is `time.day`-based, so debug clock jumps reopen stores. |
| `chopshop` | Done-untested (registered from `shops`, no own folder) | [shops.md](progress/shops.md) | `chopshop.value`, `chopshop.sell`, `chopshop.wanted`, `chopshop.setWanted:<arch>`, `chopshop.position` | `robbery` (`chop-shop-sells-sports`) | Drive-in sphere in front of the garage door, no garage interior. Registered by `ShopSystem.init` rather than discovered from `src/systems/chopshop/`. |
| `wanted` | Done-untested | [wanted.md](progress/wanted.md) | `wanted.set:<level>[:<state>]`, `wanted.clear`, `wanted.state`, `wanted.crime:<type>[:loud]`, `wanted.fastForward:<s>`, `wanted.seen`, `wanted.heat:<n>` | `wanted` | Only heat is serialized; a loaded game starts at level 0. `steal_police_vehicle` re-fires when re-entering an owned cruiser (5 s cooldown). |
| `police` | Done-untested | [wanted.md](progress/wanted.md) | `police.spawnCruiser[:police\|swat]`, `police.spawnCop[:x:z:weapon]`, `police.units`, `police.roadblock[:spikes]`, `police.heli`, `police.clear` | `police` | WASTED card/sting flashes one frame before BUSTED (UI request pending). Ramming is a predicted-position charge, not a PIT; roadblock planning fails on bends/water and retries. |
| `missions` | Done-untested | [missions.md](progress/missions.md) | `missions.list`, `missions.start:<id>`, `missions.abort`, `missions.fastForward`, `missions.complete:<id>`, `missions.unlockAll`, `missions.retry:<retry\|quit>`, `missions.retryVisible`, `missions.objective`, `missions.reset` | `missions` | m8 counter position is inferred (5.5 m past the jewelry entry), not read from the interior layout. m5 package is a sidewalk marker, not a pickup. |
| `cutscene` | Done-untested | [missions.md](progress/missions.md) | `cutscene.skip`, `cutscene.playing`, `cutscene.demo` | `missions` (m1 fast-forward runs the cutscenes) | Dialogue does not block input, so Space during a scene also jumps (accepted). |
| `companion` | Done-untested | [missions.md](progress/missions.md) | `companion.summon:<jason\|lucia>`, `companion.dismiss`, `companion.state`, `companion.hurt:<n>` | `missions` (m1/m2/m8 use `withCompanion`) | Only enters seat 1 when free; in a two-seat car with a scripted driver he stays on foot and teleport-catches-up. |
| `characters` | Done-untested | [missions.md](progress/missions.md) | `characters.switch:<lucia\|jason>`, `characters.ability`, `characters.charge[:0..1]`, `characters.state` | `characters` | Switching mid-drive dismounts first; the car is not stored per character. Jason's adrenaline uses DOM head markers, no audio pitch shift. |

Core (not a system) debug actions: `time.set:<hour>`, `time.scale:<s>`, `time.clockScale:<s>`, `time.get`, `core.systems`, `core.perf`, `core.save`, `core.load`, `core.errors`. `__game.debug.list()` prints every registered `id.action`.

---

## 2. How to test

### URL recipes

- `http://localhost:5173/?notitle` — full game, skip the title screen (also skipped whenever `systems`, `autotest` or `actions` is present).
- `?systems=world,player,camera` — isolation: only the named systems plus core. Each agent's isolation set is in its progress doc, e.g. `?systems=world,player,camera,vehicles,traffic`, `?systems=world,player,camera,economy,peds,pickups`, `?systems=world,player,camera,weapons,damage,particles`, `?systems=world,player,camera,economy,navigation,ui`, `?systems=world,player,camera,economy,interiors,casino`, `?systems=world,player,camera,audio`.
- `&spawn=x,z` — spawn position override.
- `&perf` — perf overlay on at boot (`F3` toggles it).
- `&actions=a.b:1:2,c.d` — runs debug actions after `__GAME_READY__`. Comma separates actions, colon separates args, args are JSON-parsed when possible. Consequence: an argument cannot contain `,` or `:` from the URL (so `casino.rig:slots:["diamond",...]` and `ui.objective:` with colons must be run from the console with `__game.debug.run(...)`). The `;`-separated example URL in `casino.md` will not parse; use `,` and the console for the rig.
- `&autotest=<name>` — imports `src/debug/autotests/<name>.ts`, logs `AUTOTEST PASS <name>/<check>` / `AUTOTEST FAIL <name> <reason>`. Headless: `npm run smoke -- --systems=<ids> --name=<label> --autotest=<name>` writes `docs/screenshots/<label>.png|.log` and exits non-zero on any console error or FAIL.

### Autotests (`src/debug/autotests/*.ts`)

| Name | Needs systems | Checks |
|---|---|---|
| `foundation` | `world,player,camera` (+ `economy,interiors` for those checks) | ready with zero errors; walk 20 m; jump raises player; A* on 20 random node pairs; every landmark door within 15 m of a road node; static draw calls < 400; dusk turns lamps on; water query; wallet add/spend; armor absorbs, death → respawn; interior enter/exit; camera modes + cinematic; save/load roundtrip; no errors after run. |
| `vehicles` | `world,player,camera,vehicles` | spawn sedan, enter, hold W 4 s → > 12 m/s and within 6 m of a lane; handbrake + steer → \|yaw rate\| > 0.8 rad/s; `explodeCurrent` → `vehicle:exploded`, wrecked, player ejected; SUV enter/exit leaves player alive on foot; no errors. |
| `traffic` | `world,player,camera,vehicles,traffic` | count within ±6 of target after 15 s; a car stops on red at 3 sampled lit nodes within 60 s; `traffic.stalled:20 === 0`; `traffic.panic` clears drivers within 60 m; no errors. |
| `peds` | `world,player,camera,economy,peds,pickups` | ≥ 40 peds within 8 s; aim → cower ≤ 1 s → cash pickup ≤ 3 s → scripted W collects and cash rises; fake gunshot → ≥ 50 % of peds within 40 m flee/call police, `ped:reportedCrime` ≤ 6 s; vehicle damage 90 → dead, despawned (`peds.fastForward`); 50 wanderers ≥ 90 % within 3 m of a sidewalk. |
| `weapons` | `world,player,camera,weapons,damage,particles` | pistol body shot → dummy 75; headshot kills; reload timing ±10 %; SMG 2 s → 24–30 shots; shotgun ≥ 6 pellet hits; `explode` 3 m vs 12 m falloff; Tab wheel + `time.scale === 0.2`, release equips; digits ignore unowned; `weapon:aimedAt`; 3-hit melee combo; `npcFire` hurts player; particle pools bounded; `shot_fired` crimes == shots; no errors. |
| `ui` | `world,player,camera,economy,navigation,ui` | systems present; waypoint route > 5 points ending within 15 m; landmark blips; minimap non-empty; waypoint clears on arrival; star DOM classes for each wanted state; notification shows/expires; money notification; card slow-mo restores; objective markup; prompts; progress bar; dialogue advances with Space; pause/resume; full map toggle; title flow; no errors. |
| `casino` | `world,player,camera,economy,interiors,casino` | enter → `state.interior === 'casino'`; buy 1,000 chips → cash −1,000; rigged natural at bet 100 → chips 1,150; rigged roulette 17 with $10 straight → +350; rigged diamond×3 at bet 10 → +5,000; wheel spins once, second same-day spin refused; `casino:lost` emitted; cash out + exit; no errors. |
| `audio` | `world,player,camera,audio` | system loaded; every one of 279 ids plays; 500 plays keep the voice pool bounded; engine/siren loops; radio toggle; bus volume persists; ambient bed selected; unknown id warns once; no errors. |
| `shops` | `world,player,camera,economy,interiors,weapons,peds,pickups,ui,navigation,shops,robbery` | interiors registered + chopshop present; enter Bullet Barn; buy rifle (−$3,000, owned); ammo when owned; denied when broke; armor; gun store UI open/close releases blocker; Waffle Hut food heals; safehouse save point; closed-store gate; phone open + Snapmatic data URL; hospital fee charged once. |
| `robbery` | shops set + `vehicles,wanted` | enter 24/7; aim at clerk 1 s → `robbery:started`, hands up; gunshot cuts ≥ 35 %; `robbery.skip 30` → completed, pickup +1, store closed; exit → `crime:committed rob_store` (+ `wanted.level ≥ 1`); chop shop sells a `sports` (+value, vehicle gone); respray with wanted 2 → −$2,000, level 0, colour changed. |
| `wanted` | `world,player,camera,peds,vehicles,traffic,weapons,damage,wanted,police` | witnessed shot → level 1 `responding`; cop LOS → `active`; lose LOS → `searching` then evaded; hot-scene re-entry re-raises; phone report → `responding`; crime while wanted adds stars and caps at 6; respawn clears. |
| `police` | same as `wanted` | level 4 dispatches SWAT van + helicopter (10 s budget); `clear` stands units down; helicopter takes damage and crashes (20 s budget); roadblock spawns across the road ahead; standing still next to a level-1 cop → BUSTED; no errors. |
| `missions` | `world,player,camera,ui,vehicles,peds,weapons,damage,pickups,navigation,economy,wanted,police,missions,cutscene,companion,characters` | list/unlocks; m1 `fastForward` → `mission:passed` + cash delta; m2 death → `mission:failed 'wasted'` + retry prompt, quit clears blocker; `missions.complete` m2…m8 + both jobs with zero errors; jobs available after finale; no leaked blocker / time scale / objective. |
| `characters` | `world,player,camera,ui,weapons,characters` | switch to Jason (event, state, camera mode restored, palette, stats, moved); switch back restores Lucia; adrenaline `time.scale ≈ 0.3` then 1, charge drops; recharge progresses; no errors. |

### `&actions=` examples (names taken from the code)

- `?notitle&actions=time.set:22` — night, lamps and neon on.
- `?notitle&actions=player.teleport:0:0` · `world.goto:casino` · `world.goto:prison` (landmark ids: `safehouse hospital police_downtown police_gellhorn prison casino gunstore store_habana store_gellhorn store_downtown store_beach gas_station waffle_hut jewelry bank respray chop_shop marina mud_club`).
- `?notitle&actions=vehicles.spawnHere:sports,vehicles.enterNearest` — car ready to drive.
- `?notitle&actions=weapons.give:rifle:200,weapons.equip:rifle`.
- `?notitle&actions=wanted.set:3` · `wanted.set:4` (SWAT + helicopter) · `police.roadblock:true` · `wanted.crime:shot_fired:true`.
- `?notitle&actions=missions.unlockAll` then walk to a yellow contact cylinder and press E · `missions.start:m1` · `missions.complete:m3` (fast mode).
- `?notitle&actions=casino.enter,casino.buy:5000,casino.play:blackjack`.
- `?notitle&actions=interiors.enter:gunstore` · `shops.open:phone` · `robbery.start:store_habana`.
- `?notitle&actions=characters.switch:jason` · `companion.summon:jason` · `audio.radio:true,audio.bus:music:0.4`.
- `?notitle&actions=peds.spawnHere:gang,peds.gunshot` · `traffic.setDensity:40` · `damage.explode:0:0:8:150`.
- `?notitle&actions=ui.card:wasted:Wasted` · `ui.notify:hello:money` · `navigation.waypoint:400:-300`.

### Unit tests

`npm test` → `tests/roadgraph`, `tests/spatialhash`, `tests/vehicles/handling` (11), `tests/weapons/*`, `tests/casino/*` (29 tests, ~200 assertions), `tests/audio/*` (16), `tests/shops/*` (15), `tests/wanted/machine` (15), `tests/missions/objectives` (15).

---

## 3. Cut scope / deviations from SYSTEM_PROMPT

**Foundation (world/player/camera)** — no `docs/progress/foundation.md`, `docs/CONTRACTS.md` or `docs/ARCHITECTURE.md` was written (TEAM-PLAN §5 deliverables). Swimming, weather, boats/jet ski, motorcycles, gangs-as-faction, radio-in-vehicle wiring and the juice pass (§19 Phase 10) are not present. Save/load exists per system but has not been exercised end to end in a browser.

**Vehicles / traffic** — arcade single-body fallback instead of raycast-per-wheel suspension (sanctioned by §5); yaw-locked bodies never roll over; bridge pitch is visual only. Traffic drivers are virtual (no ped mesh in the seat) until a carjack/panic spawns one; no passengers. Enter/carjack is a timed wait (0.8 / 1.2 s), no walk-to-door animation. `C` camera cycling is a no-op. No yielding at unlit intersections, no overtaking; lane changes only after 8 s blocked. No spike-strip speed penalty (spikes apply 150 damage + lateral impulse instead).

**Peds** — no crouch pose for armed gang peds; no ped-vs-ped avoidance; distant peds are un-animated rather than LOD meshes; horn reaction is a short flee, not a sidestep; idle groups have no gestures or chatter audio.

**Weapons / damage / FX** — no gun meshes or melee arm swing on the player rig; drive-by fires along camera forward with no window-side or passenger restriction; weapons are allowed inside interiors; particles unsorted and unlit.

**UI / navigation** — static map layer at 1 px/m (spec 0.5); no invert-Y / draw-distance / bloom settings (no hooks); no "Recalculating…" text and no 3D route line; dialogue does not block input (Space also jumps); no blip clustering; `wanted.lost` jingle left to the wanted system.

**Audio** — no browser listening pass (levels untuned); doppler is a loop-pitch approximation, listener velocity ignored; footstep surface from `StaticEntity.surface` only; no crowd-density chatter coupling; no weather manager (rain/wind/thunder recipes exist); no serialize (settings in localStorage).

**Casino** — roulette UI uses a rolling number display (ball animates on the 3D wheel only); slot symbols are HTML glyphs; no insurance/surrender/re-split, one seat; guards hold position and do not chase; bar is decorative; `casino.enter` teleports rather than walking through the door.

**Economy / shops / robbery / chop shop / phone** — no safehouse wardrobe recolour; jewelry/bank robbery flow belongs to missions (guards only shoot, do not report to wanted); phone feed not persisted, Snapmatic keeps only the last shot; Vice Sprays and chop shop are drive-in volumes, no garage interiors; jewelry/bank have no register cash by design.

**Wanted / police** — no distinct army archetype at 6 stars (SWAT vans/peds with rifles + 0.2 accuracy); no cover-point system (standoff + strafe); no spike speed reduction; helicopters drop no SWAT ropes; roadblocks are static; only heat survives save/load (level and units reset).

**Missions / story / characters** — no mission title card at start; no boat archetype (m8 getaway is a parked `sports` at the marina); Jason's adrenaline marks heads with DOM elements, no particles or audio pitch shift; no per-mission music.

---

## 4. Known issues (compiled, deduplicated)

**Cross-system (from the integration review)**
- Player, peds and pickups hang off `game.worldRoot`, which `Interiors.enter` hides → **everyone is invisible inside every interior** (casino, stores); shops draws clerks/guards through `PedProxy` rigs as a stopgap. Fix proposed in [shops.md](progress/shops.md) integration request 1 (`actorsRoot`). Highest priority for browser QA.
- Run-over kills (vehicle contact damage) bypass `damage.apply` → no `entity:killed` and no `kill_ped` crime.
- Hospital fee: player system charges 10 %, economy would charge 5 % min $100; only one runs, but the design number is not honoured.
- Casino station `E` and interior-exit `E` poll the same key (harmless with the current layout).
- `Player.deserialize` teleports without a ground snap; an airborne save falls one frame on Continue.
- `Vehicle.ts` (430), `Ped.ts` (~470), `WeaponSystem` (404), `CasinoSystem` (408), `MissionSystem` (~405) exceed the ~400-line guideline.
- Every autotest is unexecuted; timing tolerances are guesses.

**world / player / camera** — async city build can finish after dependent systems (`peds` waits up to 8 s for `roadGraph`). Sniper FOV kick fights the aim FOV.

**vehicles / traffic** — single ground probe under the chassis loses steering for a few frames over kerb edges; red-light queues restart with a lag chain; per-step Δv collision damage favours the moving car against sleeping parked cars until they wake; explosion fallback without `damage` does not hurt peds.

**peds / pickups** — a ped spawned on a road edge may cross mid-block when no sidewalk end is within 25 m; ragdolls jitter on moving vehicle roofs before freezing; fleeing peds can run through doorway sensor volumes.

**weapons / damage / particles** — aim ray + fire ray both run per frame while aiming and firing (accepted); sniper scope FOV recovery is the camera's, not the weapon's. (Double blood puff was fixed at integration.)

**ui / navigation** — layout never reviewed at 1440×900 / 1920×1080; Esc + pointer-lock drop in the same instant uses a 0.3 s grace window; `ui.objective:` with colons cannot be passed via URL; WASTED card duration vs death timer was fixed at integration (deathTimer divides by `time.scale`).

**audio** — loops started before the first gesture bunch their first ~0.2 s of events at resume; `play()` of a `.loop` id auto-fades after 1.5 s; listener param writes can zipper on hard camera cuts; voice stealing uses estimated loudness, not RMS.

**casino** — reopening a blackjack panel mid-deal shows cards without animation; ban kick-out shows one frame of the interior; `canSpin` compares last-spin day (intended); the doc's example URL uses `;` separators which the harness does not parse.

**economy / shops / robbery / chopshop** — clerk shotgun accuracy 0.45 at close range may kill a low-health player fast (`ShopsConfig.clerk.accuracy`); `data/weapons.ts` prices/ammo prices differ from `data/prices.ts` (store charges `prices.ts`); store interiors are 60–120 unique meshes each (diner tiles individual); guards do not report to wanted; debug clock jumps reopen closed stores; chop shop lives in `src/systems/shops/ChopShop.ts` and is registered by `ShopSystem.init`, not discovered from a `chopshop` folder.

**wanted / police** — WASTED card/sting flashes one frame before BUSTED (`Arrest.bust` kills then emits `player:busted`; UI microtask fix requested); ramming is a predicted-position charge, not a PIT; roadblock planning returns null on bends/open ground and retries; helicopter LOS ignores interiors without roof colliders; > 24 foot cops need > 1 s to sweep LOS; `steal_police_vehicle` re-fires on re-entering an owned cruiser (5 s cooldown); only heat is saved.

**missions / cutscene / companion / characters** — m8 counter position inferred, not read from the jewelry layout; m5 package is a marker, not a pickup; companion needs a free seat 1; `winAtCasino` counts any game; character switch mid-drive dismounts and leaves the car; no HUD ability meter yet (request to UI); pause menu has no "Abort mission" (optional request).

---

## 5. Definition of Done (SYSTEM_PROMPT §24)

Every item is implemented in code and typechecked; **none has been verified in a browser this session**, so nothing below can honestly be ticked. "partial" = the code path exists but a listed sub-requirement is cut or unverified; "not yet" = missing.

- [ ] **Start a New Game** — partial: title screen with New Game/Continue exists (`ui`), never clicked in a browser.
- [ ] **Watch Lucia leave prison** — partial: m1 orbit cutscene at the penitentiary gate exists (`missions`, `cutscene`); unexecuted.
- [ ] **Drive to the safehouse with GPS** — partial: m1 `goto` objective sets a waypoint; A* route + purple line exist (`navigation`, `ui`); unexecuted.
- [ ] **Steal cars from traffic** — partial: carjack (F, 1.2 s) ejects a fleeing ped and raises `carjack`; unexecuted.
- [ ] **Chased by up to six stars with helicopters and SWAT** — partial: crime table, dispatch table to level 6, helicopter, SWAT vans, roadblocks exist; no army archetype (SWAT reused); unexecuted.
- [ ] **Lose them** — partial: searching state, search circle, evasion timer, hot scenes, respray; unexecuted.
- [ ] **Shoot and rob pedestrians and stores** — partial: hitscan, cower → cash after 1.5 s, store hold-up with bag-fill bar and shotgun clerk; robbery pickup is invisible indoors until the actors-root fix; unexecuted.
- [ ] **Sell cars at the chop shop** — partial: valuation, wanted archetype ×2, drive-in volume; unexecuted.
- [ ] **Buy guns** — partial: Bullet Barn panel with weapons/ammo/armor; weapon-table price mismatch unresolved; unexecuted.
- [ ] **Walk into Leonida Palms and play blackjack, roulette, slots and the wheel with real money** — partial: all four games implemented and unit-tested (29 tests), cashier converts cash ↔ chips; player invisible inside until the actors-root fix; unexecuted in browser.
- [ ] **Complete all eight story missions with cutscenes** — partial: m1–m8 + two jobs written, objectives unit-tested (15), `missions.complete` fast mode written; never played.
- [ ] **Switch between Lucia and Jason** — partial: Q wheel, sky-up/down transition, per-character state, Caps Lock adrenaline; unexecuted.
- [ ] **Save, quit, and continue** — partial: every system has `serialize/deserialize`, safehouse bed and P quick-save, Continue on title; wanted level and police units are not saved; not exercised end to end.
- [ ] **Steady 60 fps** — not yet: no measurement exists (see §6); no performance pass done.
- [ ] **Stylized Vice City at sunset that turns into a neon night** — partial: 24-minute cycle, lamps/neon at dusk, `dusk-turns-lamps-on` autotest; no screenshot reviewed this session.
- [ ] **`PROGRESS.md` accurately describes what exists** — done to the extent the per-system docs are accurate; will need revision after the first browser run.

---

## 6. Performance budget (SYSTEM_PROMPT §20)

| Metric | Target | Measured |
|---|---|---|
| Frame rate, M1 MacBook Air 1440×900 | 60 fps | not yet (needs browser run) |
| Frame rate, Intel iGPU, shadows off | never below 30 fps | not yet (needs browser run) |
| Draw calls | ≤ 900 total (< 400 static, Phase 1) | not yet (`foundation` autotest asserts < 400 static; `core.perf` / F3 report it) |
| Visible triangles | ≤ 1.5 M | not yet |
| Shadow-casting lights | 1 directional, 2048 map, 120 m bound following the player | not yet (code claims one directional light; unverified) |
| Dynamic point/spot lights on at once | ≤ 8 | not yet (known sources: headlights, muzzle flash light, helicopter spotlight at night, 5 casino point lights, store point lights — casino + store interiors may exceed 8 while indoors) |
| Active physics bodies near player | ≤ 250; peds beyond 60 m visual-only | not yet (peds 60–150 m step every 3rd tick, beyond 150 m no physics; parked cars sleep) |
| Per-frame allocations in `update()` after warmup | zero; pools for peds, vehicles, particles, pickups, tracers, audio voices | not yet measured; code review found only transition-time allocations |
| Perf overlay | F3 with fps, draw calls, triangles, bodies, peds, cars | present (`Debug.togglePerf`, `&perf`); unverified |
