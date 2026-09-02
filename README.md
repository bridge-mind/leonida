# LEONIDA

A GTA VI-style open-world crime sandbox that runs in the browser. Vice City, State of Leonida, rebuilt as a
stylized low-poly city in Three.js with Rapier physics. Everything — buildings, cars, people, audio — is generated
from code; there are no downloaded assets.

Non-commercial fan project for testing. No Rockstar assets, logos, fonts or music are used. Grand Theft Auto, GTA and
related names are trademarks of Take-Two Interactive; this project is not affiliated with or endorsed by Rockstar Games
or Take-Two.

## Play it

**https://bridge-mind.github.io/leonida/** — built from `main` on every push by [GitHub Actions](.github/workflows/deploy.yml).
Desktop Chrome, Edge or Firefox with WebGL2 and a keyboard + mouse; no mobile support.

Found a bug? [Open an issue](https://github.com/bridge-mind/leonida/issues) with the URL you used, the console output
and the results of the debug actions listed under [Debugging](#debugging).

## Run it

```bash
npm install
npm run dev        # http://localhost:5173
```

Click the canvas to lock the mouse. `Esc` pauses. Full controls in [CONTROLS.md](CONTROLS.md).

Other scripts:

```bash
npm run typecheck  # tsc --noEmit
npm test           # vitest (casino rules, road graph, spatial hash, audio pool ...)
npm run build      # production bundle in dist/
npm run preview    # serve dist/
npm run smoke      # headless boot check (writes docs/screenshots/<name>.png + .log)
```

## What is in the game

See [docs/PROGRESS.md](docs/PROGRESS.md) for the authoritative, per-system status (works / cut / known bugs / how to test). Current build: typechecked, unit-tested, code-reviewed and boots headlessly with zero console errors (`npm run smoke`), but it has had **very little hands-on play-testing** — expect rough edges and please report them.

- **World**: 2 km × 2 km Vice City with six districts (Vice Beach, Downtown, Little Habana, Port Gellhorn, the Docks, Grassrivers), a causeway, highways, water, and a 24-minute day/night cycle with neon and street lamps at dusk. No swimming, weather or boats.
- **On foot**: third-person GTA camera, sprint, jump, aim, health/armor, WASTED → hospital.
- **Vehicles**: nine archetypes (sedan, sports, muscle, SUV, pickup, van, taxi, police, SWAT), arcade single-body physics on Rapier (no per-wheel suspension), damage → smoke → fire → explosion, carjacking, traffic with lights and car-following, parked cars everywhere. No motorcycles or boats.
- **Pedestrians**: instanced crowds that wander, chat, cower, flee, call the cops, fight back, and ragdoll; cash/weapon/armor/health pickups.
- **Combat**: six weapons (fists, pistol, SMG, shotgun, rifle, sniper), weapon wheel, hitscan with headshots, melee combo, particles and tracers. No gun meshes on the player rig.
- **Wanted**: six stars with the GTA 6 colored states (responding / searching / active), hot crime scenes, cruisers, cops on foot, SWAT, helicopters, roadblocks with spike strips, BUSTED → police station, respray to clear. Level 6 reuses SWAT (no army).
- **Money**: rob peds and stores (24/7 ×4, gas station, Waffle Hut), sell cars at the chop shop, buy guns at Bullet Barn, eat, gamble. Chop shop and respray are drive-in markers, not garages.
- **Casino**: Leonida Palms — blackjack (full rules, unit-tested), American roulette, slots, Lucky Wheel, cashier, guards.
- **Story**: eight missions for Lucia and Jason with cutscenes, two repeatable jobs after the finale, an AI companion, Q character switching and the Caps Lock adrenaline ability.
- **Meta**: minimap and full map with GPS routing, HUD, notifications, pause menu with settings, phone (contacts, map, Snapmatic, parody feed), per-system save/load (wanted level and police units are not saved).

Known first-run blocker: the player, peds and pickups are currently invisible inside interiors (they are parented to the hidden world root); see `docs/PROGRESS.md` §4.

## Debugging

- `F3` toggles the performance overlay (`&perf` in the URL turns it on at boot).
- `window.__game` is the `Game` instance; `__game.debug.list()` prints every debug action; `__game.debug.run('vehicles.spawnHere', 'sports')`. Core actions: `time.set:<hour>`, `time.scale:<s>`, `core.systems` (which systems are enabled/disabled), `core.perf`, `core.errors` (collected console errors), `core.save`, `core.load`.
- URL params: `?systems=world,player,camera,vehicles` loads an isolation set; `&actions=time.set:22,player.teleport:0:0` runs actions after boot (comma between actions, colon between args, args JSON-parsed when possible, so arguments cannot contain `,` or `:`); `&autotest=<name>` runs `src/debug/autotests/<name>.ts` and logs `AUTOTEST PASS/FAIL`; `&notitle` skips the title screen (also skipped when `systems`, `autotest` or `actions` is present); `&spawn=x,z`.
- A system that throws is logged as `[system:<id>]` and disabled for the session; the game keeps running. Check `__game.debug.run('core.systems')` when something silently stops working.
- Reporting a bug: the URL you used, the console output, and the results of `core.errors`, `core.systems` and `core.perf`. Full recipes and every system's debug actions are in [docs/PROGRESS.md](docs/PROGRESS.md) §2.

## Layout

```
src/core       game loop, events, input, time, save, state, contracts (peers.ts)
src/physics    Rapier wrapper
src/world      road graph, city generator, districts, landmarks, water, day/night, interiors
src/entities   Player, Vehicle, Ped, Pickup
src/systems    one folder per system (world, player, camera, vehicles, traffic, peds, ...)
src/ui         HTML/CSS overlays
src/casino     pure game logic (unit-tested)
src/missions   story missions
src/data       tunable tables (weapons, vehicles, peds, prices, landmarks, dialogue)
docs           TEAM-PLAN, PROGRESS, per-system progress notes, screenshots
```

## License

Code is released under the [MIT License](LICENSE). Character names and the setting are fan-fiction references to an
unreleased Rockstar Games title and remain the property of their respective owners.
