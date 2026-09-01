# GTA 6 → Three.js: What It Is, What We Can Build, and How

Written 2026-09-01. Companion to `SYSTEM_PROMPT.md` in this folder (the prompt handed to the build agent).

---

## 1. What GTA 6 actually is (public information as of Sept 2026)

- **Publisher/status:** Rockstar Games. Releases **Nov 19, 2026** on PS5 / Xbox Series X|S ($79.99 / $99.99 Ultimate). No PC date announced. Two trailers (Dec 2023, May 2025) plus an "Extended Look" gameplay reveal.
- **Setting:** the **State of Leonida** (Florida) centered on **Vice City** (Miami). Confirmed regions: Vice Beach (art-deco Ocean Drive strip, neon nightlife, pool parties), downtown skyline, **Port Gellhorn** (working-class coastal town), the Keys, **Grassrivers** (Everglades-style swamp with airboats and gators), Mud clubs and trailer parks, and highways connecting it all.
- **Protagonists:** **Lucia Caminos** (starts the game fresh out of Leonida Penitentiary) and **Jason Duval**. A Bonnie-and-Clyde couple. **Dual-protagonist control:** a character wheel offers Solo Jason, Solo Lucia, or "both" – one drives while the other shoots, and you can switch mid-robbery.
- **Special abilities:** both have slow-motion abilities (Jason: multi-target, highlights weak points like RDR2 Dead Eye; Lucia: a single precision shot).
- **Wanted system:** **six stars** (up from five in GTA V). Star **colors** carry meaning: **black w/ white outline** = police responding to a crime scene but don't know it's you; **white** = they know a crime happened and are searching; **red** = active pursuit. Losing the cops no longer fully clears you: the **crime scene stays hot on the map** and returning can reactivate your wanted level. A **Criminal Profile** system judges crimes with nuance (returning fire on an aggressor ≠ a killing spree).
- **Robberies** are core: gas stations, a waffle restaurant, and larger jobs. Face coverings live in an item wheel. Jason and Lucia can rob a place together.
- **Phone / social media:** Snapmatic returns; an optional parody livestream feed (R3); heavy social-media satire (influencers, mud-club streamers, etc.).
- **Body & relationships:** food affects weight, exercise builds muscle (RDR2-style). NPCs have ages, body types, tattoos, and relationships with the player.
- **Activities:** swimming and scuba, kayaks, jet skis, dirt bikes, base jumping, basketball, weightlifting, strip clubs and nightclubs.
- **Casino:** not confirmed for GTA 6 single-player, but the user wants it and GTA Online's **Diamond Casino** defines the template: blackjack, roulette, slots, three-card poker, the Lucky Wheel, and Inside Track horse racing, all played with chips bought 1:1 with cash.

Everything not listed above (exact map, mission list, radio stations) is unknown, so we fill gaps with the **GTA formula** that every mainline entry shares.

## 2. The GTA formula (why it is fun)

The loop is **freedom with consequences**:

1. **Move through a living city** on foot or in any vehicle you can steal.
2. **Do crime for money:** rob pedestrians and stores, steal and sell cars, run missions, pull heists.
3. **Escalating police response** is the risk; **evasion** is the tension.
4. **Spend money on power and play:** guns, armor, cars, safehouses, the casino, clothes.
5. **Story missions give direction; the sandbox gives emergent chaos.**

The magic is that peds, traffic, cops, and physics all **react to each other** through shared systems. A fender-bender becomes a shootout becomes a six-star chase without any scripting. That systemic reactivity, not visual fidelity, is what we have to reproduce.

## 3. System decomposition: real game → our clone

| System | What GTA 6 does | What our Three.js version does |
|---|---|---|
| World | ~Florida-sized state, hand-built | 2 km × 2 km stylized low-poly Vice City + swamp edge, procedurally generated from a road graph with hand-placed landmarks, 6 districts |
| On-foot | Mocap animation, cover, climbing, swimming | Kinematic capsule with GTA third-person camera, sprint/jump, aim-over-shoulder, procedural limb animation; swimming as a later phase |
| Vehicles | Hundreds of models, damage deformation, radio | 10 procedural box-built archetypes (sedan, sports, muscle, SUV, pickup, van, taxi, cruiser, SWAT van, bike/boat later), Rapier raycast-vehicle physics, damage → smoke → fire → explosion, procedural engine audio |
| Traffic | Full lane simulation | Path-following kinematic cars on lane splines with timed intersections and car-following; convert to dynamic bodies when hit |
| Peds | Lifelike NPCs, relationships | 60–100 pooled, instanced blocky peds with a state machine: wander, idle, flee, cower (hands up), fight, call-police, ragdoll |
| Combat | Weapon wheel, free/soft aim, cover | Hitscan raycasts, 7 weapons, weapon wheel, headshots, ragdoll deaths, muzzle flash/tracer/impact FX, optional soft aim-assist |
| Wanted | 6 stars, colored star states, hot crime scenes, criminal profile | Same 6-star model with responding/searching/active states, search circle on minimap, hot scenes, roadblocks, helicopter, SWAT, respray to clear |
| Money | Cash and everything it buys | Wallet + rob peds/stores, chop shop car sales, missions, casino; spend at gun store, respray, casino, safehouse |
| Casino | Diamond Casino (Online) | Interior scene with cashier, full-rules blackjack, American roulette, 3-reel slots, daily Lucky Wheel (poker & Inside Track optional) |
| Map | Minimap + full map + GPS | Canvas minimap that rotates with heading, full-screen pan/zoom map, click-to-set waypoint, A* GPS route drawn on the road graph |
| Missions | Story with cutscenes | 8-mission Lucia & Jason arc (prison release → first robbery → chop shop → debt → swamp ambush → casino → heat → heist finale), scripted with objective lists, letterbox cutscenes, text dialogue |
| Dual protagonists | Solo/duo control, mid-mission switching | Both characters exist in the world; switch with a key; the other acts as an AI companion in missions (later phase) |
| Abilities | Slow-mo | One adrenaline slow-mo ability (later phase) |
| Phone/social | Snapmatic, livestream, contacts | Phone overlay with contacts (start missions), map, Snapmatic screenshot, parody social feed that reacts to your crimes (later phase) |
| Audio | Licensed radio, VO | Procedural Web Audio (engines by RPM, sirens, gunshots, ambience); no VO, no licensed music |
| Persistence | Save slots | localStorage JSON save/load, autosave after missions |

## 4. What is feasible in Three.js, and what is not

**Feasible at 60 fps in a browser:** a 4 km² low-poly city, physically driven cars, hitscan combat, ~80 peds and ~30 traffic cars near the player, police pursuit AI, canvas UI, casino games, save/load. Instancing and pooling make this comfortable on an M-series laptop.

**Not feasible (and not the point):** photorealism, motion-captured animation and facial performance, a licensed soundtrack, a 100 km² map, voice acting. We go **stylized low-poly** ("PS2-era GTA meets Synty POLYGON") and win on **systems and feel**: camera lag, car weight, screen shake, the money counter ticking up, stars flashing when you escape.

## 5. Key technical decisions (and why)

1. **Vite + TypeScript + three r185 + Rapier 0.20 (`@dimforge/rapier3d-compat`).** Vite is zero-config; TS keeps a large agent-written codebase honest; the Rapier compat build embeds the WASM so no bundler plugins are needed. Current npm versions checked 2026-09-01.
2. **Everything is procedural. No asset downloads.** Buildings from boxes with canvas-generated window textures, cars from box compositions, peds from six instanced parts, audio synthesized. The build agent can't reliably source models, and procedural content keeps the whole game reproducible from source. A `public/models/` drop-in path with fallback lets the user add GLBs later.
3. **Hybrid vehicle physics.** The player's car and police cars use Rapier's raycast vehicle controller (real suspension, drifting, ramming). Traffic is kinematic path-followers that flip to dynamic bodies on contact. This keeps 30 traffic cars cheap while collisions still feel physical. Explicit fallback: if the vehicle controller can't be made to feel good, a custom arcade car on a kinematic body is acceptable.
4. **The road graph is the spine.** Traffic lanes, sidewalks, police pathfinding, GPS routing, spawn points, and the minimap all derive from one `RoadGraph` data structure. Get it right in Phase 1 and every later system is simpler.
5. **Event bus for crimes.** Systems emit `crime` events with a position and type; the wanted system decides whether a witness or cop saw it. This decouples weapons, vehicles, robbery, and police.
6. **UI is HTML/CSS overlays, not 3D.** HUD, weapon wheel, map, casino tables, phone, menus. Far faster to build and iterate; matches how GTA's UI is flat anyway. Minimap is a 2D canvas drawn from map data.
7. **Interiors are separate mini-scenes** (casino, gun store, 24/7, safehouse) entered via door markers with a fade. Avoids building real interiors inside collision shells.
8. **Strict phased delivery.** The game must run and be playable at the end of every phase, with acceptance criteria the agent verifies in a browser before moving on.

## 6. Scope tiers

- **Must (a real GTA loop):** city, drive/steal cars, traffic, peds, guns, rob peds and stores, 6-star wanted with cops, minimap/map/GPS, HUD, money, gun store, casino (blackjack, roulette, slots, wheel), 8 missions, save/load, day/night.
- **Should:** helicopters, SWAT, roadblocks, respray shop, chop shop, hot crime scenes, phone, dual-protagonist switching, slow-mo ability, boats/jet ski, weather.
- **Could:** motorcycles/dirt bikes, swimming/scuba, AI-companion heist finale, gangs, clothes shop, properties, parody social feed, Snapmatic, Inside Track, three-card poker, body/skill progression, first-person camera, cover system.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Vehicle handling feels bad (flipping, jitter) | Lowered center of mass, tuned defaults provided, explicit arcade-model fallback |
| Frame rate collapses with peds/traffic | Per-part InstancedMesh, distance LOD, pooling, spatial hash, a hard draw-call budget |
| Agent rabbit-holes on one feature | Phases with acceptance criteria and time-boxing rules; "playable at every step" |
| Rapier API drift | Pinned version, the exact calls the agent needs are listed in the prompt |
| Scope creep | Must/Should/Could tiers; Must is finished before any Should |
| Silent stubs | Prompt forbids fake implementations; a `PROGRESS.md` ledger records what is real |

## 8. Legal note

This is a non-commercial fan project for testing. Do not use Rockstar assets, logos, music, or the Pricedown font. Lucia and Jason are used as the user requested; supporting cast, businesses, and radio are invented. Suggested working title: **LEONIDA**.

## 9. How to use the system prompt

1. The build agent runs inside `gta6/` (this folder), which matches how the sibling projects in the parent directory are set up.
2. Launch it with the prompt as its system prompt, e.g. `cd gta6 && claude --append-system-prompt-file SYSTEM_PROMPT.md`, or copy `SYSTEM_PROMPT.md` to `CLAUDE.md`.
3. Kick off with: **"Read SYSTEM_PROMPT.md and begin at Phase 0. Work through all phases autonomously, verifying each in the browser before moving on. Keep PROGRESS.md current."**

Sources consulted for the GTA 6 facts above:
- https://www.gtabase.com/grand-theft-auto-6/
- https://www.gtaboom.com/grand-theft-auto-vi-a428
- https://www.dexerto.com/gta/gta-6-confirms-realistic-npcs-deeper-relationships-body-changes-social-media-system-more-3402704/
- https://www.keengamer.com/articles/news/gta-6-extended-look-recap-all-gameplay-details/
- https://www.engadget.com/2246846/things-you-may-have-missed-gta-6-extended-look/
- https://www.gta6-news.net/ga6-gameplay-features/gta-6-extended-look-hides-huge-gameplay-details-from-six-star-wanted-levels-to-relationship-system/
- https://www.gtaboom.com/gta-6-countdown
