# LEONIDA

A GTA VI-style open-world crime sandbox that runs in the browser. Vice City, State of Leonida, rebuilt as a
stylized low-poly city in Three.js with Rapier physics. Everything — buildings, cars, people, audio — is generated
from code; there are no downloaded assets.

Non-commercial fan project for testing. No Rockstar assets, logos, fonts or music are used.

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

See [docs/PROGRESS.md](docs/PROGRESS.md) for the authoritative, per-system status (works / cut / known bugs / how to test).

- **World**: 2 km × 2 km Vice City with six districts (Vice Beach, Downtown, Little Habana, Port Gellhorn, the Docks, Grassrivers), a causeway, highways, water, and a 24-minute day/night cycle with neon and street lamps at dusk.
- **On foot**: third-person GTA camera, sprint, jump, aim, health/armor, WASTED → hospital.
- **Vehicles**: nine archetypes, arcade physics on Rapier bodies, damage → smoke → fire → explosion, carjacking, traffic with lights and car-following, parked cars everywhere.
- **Pedestrians**: instanced crowds that wander, chat, cower, flee, call the cops, fight back, and ragdoll.
- **Combat**: six weapons, weapon wheel, hitscan with headshots, melee combo, particles and tracers.
- **Wanted**: six stars with the GTA 6 colored states (responding / searching / active), hot crime scenes, cruisers, cops on foot, SWAT, helicopters, roadblocks, BUSTED → police station, respray to clear.
- **Money**: rob peds and stores, sell cars at the chop shop, buy guns at Bullet Barn, eat, gamble.
- **Casino**: Leonida Palms — blackjack (full rules, unit-tested), American roulette, slots, Lucky Wheel.
- **Story**: eight missions for Lucia and Jason with cutscenes, an AI companion, character switching and the adrenaline ability.
- **Meta**: minimap and full map with GPS routing, HUD, notifications, pause menu, phone, save/load.

## Debugging

- `F3` toggles the performance overlay.
- `window.__game` is the `Game` instance; `__game.debug.list()` prints every debug action; `__game.debug.run('vehicles.spawnHere', 'sports')`.
- URL params: `?systems=world,player,camera,vehicles` loads an isolation set; `&actions=time.set:22,player.teleport:0:0` runs actions after boot; `&autotest=<name>` runs `src/debug/autotests/<name>.ts`; `&notitle` skips the title screen; `&spawn=x,z`.

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
