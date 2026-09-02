# Agent 9 — Economy, Shops, Robberies, Chop Shop, Phone

Stage 3. Owned paths: `src/systems/economy/**`, `src/systems/shops/**`, `src/systems/robbery/**`, `src/systems/interiors/**`, `src/world/interiors/**` (except `Interiors.ts`, `CasinoInterior.ts`), `src/data/prices.ts`, `src/data/shops.ts`, `src/ui/phone/**`, `src/ui/shops/**`, `src/debug/autotests/shops.ts`, `src/debug/autotests/robbery.ts`, `tests/shops/**`.

Verified with `npm run typecheck` (zero errors in owned files) and `npx vitest run tests/shops` (15 tests). Autotests were **not** run this session (browser automation forbidden) — see Known issues.

## What works

### Data
- `src/data/prices.ts`: weapon prices + ammo per clip (pistol 500/30, smg 1200/60, shotgun 1500/80, rifle 3000/100, sniper 5000/150), armour 500 (+100), food (hot dog 10/+25, waffle 15/+50, snack 5/+15), respray 1000/star min 1000 (`resprayCost`), chop valuation (`chopHealthMultiplier`: ×1 at 100 % health → ×0.4 at 20 %, 0 below; `chopValue` rounds to $10, ×2 for the wanted archetype).
- `src/data/shops.ts`: 12 `StoreDef`s keyed by landmark id — `gunstore`, `store_habana`, `store_gellhorn`, `store_downtown`, `store_beach`, `gas_station`, `waffle_hut`, `safehouse`, `jewelry`, `bank` (interiors) and `respray`, `chop_shop` (drive-in volumes). Fields: kind, `registerCash` (200–800 for robbable stores, `null` for jewelry/bank/safehouse), `clerkType`, `shotgunChance` (0.25), `food`, `guards`, `hasInterior`.

### EconomySystem (`src/systems/economy`)
- API unchanged: `cash`, `add`, `trySpend`, `canAfford`; serialize shape extended (`fromRobberies`, `fromChopShop`) but backwards compatible.
- `fee(fraction, reason, min)`. Hospital bill 5 % (min $100) on `player:respawned {reason:'wasted'}` — **only if no `'hospital'` charge happened this frame** (the player system currently bills 10 % itself; whichever the integrator keeps, the fee is never charged twice).
- Stats published to `state.stats`: `cashEarned`, `cashSpent`, `moneyFromRobberies` (from `robbery:completed`), `moneyFromChopShop` (from `chopshop:sold`).
- Debug: `economy.add/set/get`, `economy.fee(fraction, reason)`, `economy.stats`.

### Interiors (`src/systems/interiors`, `src/world/interiors`)
- `InteriorsSystem` additions: door prompt shows `<name> (closed)` + `store.closed` sound and refuses entry when `game.shops.isStoreClosed(id)`; `toWorld(local, out)`; `fadeThrough(atBlack, hold)` (shared fade for respray / chop shop, same `#fade` overlay as the framework).
- `StoreKit.ts`: shared primitives (room shell with door gap, shelves, counter + register, canvas signs, point lights, cached materials/geometries, static colliders in world space). Every store is offset along +X by `STORE_SLOT_SPACING × slot` (slot 0 = casino) so colliders at the shared interior origin never overlap.
- Interiors, one file each, registered through `interiors.register` by `ShopSystem` on `game:ready`: `GunStoreInterior` (counter, glass case with pistols, wall racks with weapon-silhouette boxes, ammo crates, armour mannequin), `ConvenienceInterior` (24/7 ×4 and the gas station variant: aisles, fridges, hot-dog roller, ATM), `DinerInterior` (Waffle Hut: checkerboard floor, counter with stools, back bar, three booths), `SafehouseInterior` (bed = save point, wardrobe, couch/TV, kitchenette), `GuardedInterior` (jewelry: display cases + chandelier; bank: teller counter with bars, rope line, vault door; two guard posts each).
- Each interior exposes a `StoreLayout` (counter stand, clerk post, register point, guard posts, bed) that the systems use for prompts, ped placement and the cash pickup.

### ShopSystem (`src/systems/shops`, id `shops`, implements `IShopsPeer`)
- Staffs stores on `interior:entered` (clerk with `ClerkBrain`, guards with `GuardBrain`, via `game.peds.spawn`) and despawns on exit. Peds get a visible proxy rig inside the interior (`PedProxy`) — see Known issues for why.
- Counter prompts (`[E]`): Bullet Barn → `Browse weapons` opens the store UI; 24/7 / gas / Waffle Hut → `Buy <food> $X` heals; safehouse bed → `Save game` (`game.save.save()` + `save.complete`).
- `Purchases`: buy weapon (one clip), buy ammo once owned (one clip), armour; `economy.trySpend` → `weapons.give/addAmmo` / `player.addArmor`; emits `shop:purchased {itemId, price}`, plays `shop.buy` / `shop.denied`.
- `GunStorePanel` (`src/ui/shops/GunStore.ts` + `shops.css`): modal like the pause menu (`input.pushBlocker('shop')`, pointer released, relocked on close), rows per weapon + armour, arrows/WASD + Enter, Backspace/E/Esc to leave, live cash header.
- Clerk: `shotgunChance` is rolled once per visit. Armed clerk pulls a shotgun (`ped.weaponId='shotgun'`, fires via `weapons.npcFire` every 1.1 s) when the player keeps a gun out for > 2 s without aiming at him, or — during a hold-up — after 3 s if the aim leaves him for > 0.5 s. Unarmed clerk shot at → hands up.
- Guards (jewelry/bank): face the player; drawn weapon → "Drop it" notification, hostile after 0.8 s (pistol, 0.75 s interval); also hostile on being shot or hearing gunfire. Emits private `shops:hostile`.
- `isStoreClosed(id)` until the next in-game day after a robbery; persisted (`serialize` = `{closed: {id: untilDay}}`). Blip labels get `(closed)`.
- Hosts the phone and Vice Sprays (`Respray.ts`) and registers the chop shop system.
- Debug: `shops.open(gunstore|phone)`, `shops.closeUi`, `shops.buy(<weaponId>|armor)`, `shops.food`, `shops.close(storeId)`, `shops.reopen(storeId)`, `shops.list`, `shops.status`, `shops.armClerk(true|false)`.

### RobberySystem (`src/systems/robbery`, id `robbery`, implements `IRobberyPeer`)
- Inside a robbable store, `weapon:aimedAt {target: clerk}` with `weapons.current !== 'fists'` for 1.0 s continuous → `start()`: clerk hands up, `ui.progressBar.show('Filling bag', f)` over 12 s, `robbery:started`, register/bag-fill sounds, clerk lines via notifications.
- Player `weapon:fired` inside cuts the remaining time by 40 % (max 2×) and marks the crime loud. Pure timing model in `progress.ts` (`RobberyClock`, unit-tested).
- Completion: cash pickup (`pickups.spawn('cash', registerPoint, amount)`, amount from `registerCash` rounded to $10), `robbery:completed`, store closed.
- Leaving the interior after a hold-up started → `crime:committed {type:'rob_store', loud}` once (wanted is never touched directly).
- `rob_ped`: `CivilianBrain` already emits `crime:committed rob_ped` when a civilian is aimed at and drops cash — this system does **not** emit it (no duplication).
- Debug: `robbery.start(storeId)`, `robbery.status`, `robbery.skip(seconds)`, `robbery.abort`.

### Chop shop (`src/systems/shops/ChopShop.ts`, id `chopshop`, implements `IChopShopPeer`)
- Registered into `game.systems` by `ShopSystem.init` (see Known issues — no `src/systems/chopshop` folder). Reachable as `game.systems.get<IChopShopPeer>('chopshop')`.
- Volume 5 m outside the Docks garage door (radius 6). Player vehicle inside at < 1 m/s → `[E] Sell for $X (wanted 2×)`; police vehicles refused; wrecks below 20 % health refused.
- Sell: fade → `vehicles.despawn(v)` (force-exits the player at the door) → `economy.add(value, 'chopshop')` → `chopshop:sold`, `chopshop.crush/paid`, notification.
- `wantedArchetype`: deterministic per in-game day (`Random.fromString('chopshop:<day>')`, excludes police/swat), announced on `time:dayChanged`, shown in the chop-shop blip label.
- Debug: `chopshop.value`, `chopshop.sell`, `chopshop.wanted`, `chopshop.setWanted(id)`, `chopshop.position`.

### Respray (`src/systems/shops/Respray.ts`)
- Volume 4.5 m outside the Vice Sprays door (radius 5.5). Player vehicle stopped inside → `[E] Respray $X` (`resprayCost(wanted.level)`).
- Refuses with "They can see you — lose them first." when `wanted.playerSeen`; otherwise `trySpend`, fade 1 s, random palette colour (different from current), `wanted.clear('respray')`, `respray.done`, private `respray:done`.
- Debug: `respray.spray`, `respray.position`.

### Phone (`src/ui/phone`)
- `ArrowUp` opens (when no blocker / modal UI); modal like the pause menu (blocker `phone`, pointer released). Arrows/Enter navigate, Backspace back, Esc closes. Bottom-right panel with clock/day/cash.
- Apps: **Contacts** (`game.missions.available()`, select → "Call" → `missions.start(id)`; "No contacts yet" otherwise), **Map** (closes the phone and runs `ui.map`), **Snapmatic** (renders one frame then `renderer.domElement.toDataURL('image/png')`; thumbnail + "Save" download; `lastDataUrl` kept for tests), **Feed** (parody posts from `crime:committed` (player), `casino:won ≥ 500`, `economy:cashChanged |Δ| ≥ 2000`, `robbery:completed`, `chopshop:sold`, `mission:passed`; last 20, fake handles, not saved).
- Debug: `phone.open`, `phone.close`, `phone.snap` (returns the data URL), `phone.app(id)`, `phone.feed`.

### Blips
- Navigation already creates one blip per landmark (kinds `gunstore`, `store`, `gas`, `food`, `safehouse`, `respray`, `chopshop`, `jewelry`, `bank` exist in `blipStyles.ts`), so nothing new is registered; the shop system relabels store blips (`(closed)`) and the chop shop blip (`— wants <archetype>`). No new blip kinds needed.

## How to test

- Unit: `npx vitest run tests/shops` — `prices.test.ts` (price table, respray cost, chop valuation, store table), `robbery.test.ts` (`RobberyClock`).
- Autotest `shops` (`src/debug/autotests/shops.ts`): interiors registered + chopshop system present → enter Bullet Barn → `shops.buy rifle` (−$3000, `weapons.has('rifle')`) → ammo purchase → refused when broke → armour → gun store UI open/close (blocker released) → Waffle Hut food heals → safehouse layout → closed-store gate → phone open/snap (`data:image/png`)/feed/close → hospital fee charged exactly once.
  `npm run smoke -- --systems=world,player,camera,economy,interiors,weapons,peds,pickups,ui,navigation,shops,robbery --name=shops --autotest=shops`
- Autotest `robbery` (`src/debug/autotests/robbery.ts`): enter `store_habana`, find the clerk via `peds.query(typeId==='clerk')`, equip pistol, emit `weapon:aimedAt` each frame → `robbery:started` + clerk hands up → `weapon:fired` cuts ≥ 35 % → `robbery.skip 30` → `robbery:completed` + pickup count +1 + store closed → exit → `crime:committed rob_store` (and `wanted.level ≥ 1` if the wanted system is loaded) → chop shop: spawn `sports` at `chopshop.position`, `vehicles.enterNearest`, `chopshop.sell` → +value, vehicle gone, `chopshop:sold` → respray: `wanted.setLevel(2)` if present, sedan in the volume, `respray.spray` → −$2000 (−$1000 without wanted), level 0, colour changed.
  `npm run smoke -- --systems=world,player,camera,economy,interiors,weapons,peds,pickups,vehicles,ui,navigation,shops,robbery,wanted --name=robbery --autotest=robbery`
- Manual: walk into any store door (`[E] Enter …`), `shops.status` in the console, `ArrowUp` for the phone.

## Integration requests

1. **Actors are invisible inside interiors (foundation, high priority).** `Interiors.enter` sets `game.worldRoot.visible = false`, but the player rig (`src/entities/Player.ts:82`), the ped renderer (`src/systems/peds/index.ts:74`, `new PedRenderer(game.worldRoot)`) and the pickup renderer (`src/systems/pickups/index.ts:35`) all hang off `worldRoot`, so inside the casino and every store the player, clerks, guards and the robbery cash pickup are not drawn. Proposed fix — `src/core/Game.ts`: add an actors root next to the world/interior roots and re-parent the three renderers to it:
   ```ts
   // src/core/Game.ts
   readonly worldRoot = new Group();
   readonly interiorRoot = new Group();
   /** Player, peds, pickups: visible both in the city and inside interiors. */
   readonly actorsRoot = new Group();
   // constructor:
   this.scene.add(this.worldRoot, this.interiorRoot, this.actorsRoot);
   ```
   ```ts
   // src/entities/Player.ts (constructor / dispose)
   game.actorsRoot.add(this.object3d);      // was game.worldRoot.add
   this.game.actorsRoot.remove(this.object3d);
   // src/systems/peds/index.ts
   this.renderer = new PedRenderer(game.actorsRoot);
   // src/systems/pickups/index.ts
   this.renderer = new PickupRenderer(this.game.actorsRoot);
   ```
   Once applied, delete `src/systems/shops/PedProxy.ts` and the `proxies` handling in `StoreRuntime.ts` (otherwise clerks render twice).
2. **Hospital fee ownership.** `src/systems/player/index.ts:165` bills `Config.player.deathCashFraction` (10 %) as `'hospital'` on respawn. Design says 5 % min $100 via economy. Economy already skips its fee when the player system charged this frame, so either remove that line (`if (loss > 0) economy.trySpend(loss, reason === 'wasted' ? 'hospital' : 'bail');` → keep only the `'busted'` branch) or leave it — no double charge either way.
3. **Escape while a modal panel is open** — `src/systems/ui/index.ts:297`: `if (input.wasPressed('Escape')) { this.pause(); return; }` also fires while the casino / gun store / phone have a blocker pushed, so Esc closes the panel *and* opens the pause menu. Suggested: `if (input.wasPressed('Escape') && !input.isBlocked) {`.
4. **`chopshop` system module.** `Game.start` only discovers `src/systems/*/index.ts`, and `src/systems/chopshop/` is outside this agent's paths, so `ShopSystem.init` adds `CHOPSHOP_MODULE` to the registry itself. If a dedicated folder is preferred, create `src/systems/chopshop/index.ts` with `export { CHOPSHOP_MODULE as default } from '../shops/ChopShop';` — `registerChopShop()` already skips when `systems.has('chopshop')`.
5. **Weapon table prices** — `src/data/weapons.ts` carries `price`/`ammoPrice` that differ from the design table (smg 1500/60 vs 1200/60, shotgun 1200/50 vs 1500/80, rifle ammo 80 vs 100, sniper ammo 100 vs 150). `src/data/prices.ts` is what the store charges; either sync the weapon table to it or drop those two fields so there is one source of truth.

## Known issues

- **Autotests not executed** this session (browser automation forbidden). Both are written against the debug/event APIs only; the first run may need timing tweaks (waits are generous).
- Peds/pickups/player invisible indoors until integration request 1 lands; clerks and guards are drawn through `PedProxy` rigs (position/heading/pose mirrored per frame). The robbery cash pickup is invisible indoors but collectable (walk up to the register).
- (resolved) `ArrowUp` no longer moves the player; the phone consumes the key.
- Esc closes the gun store / phone but the UI system still opens the pause menu on the same key (request 3). Backspace / E / the home button are the clean way out.
- Clerk shotgun fire uses `weapons.npcFire` accuracy 0.45 at close range — may be lethal fast for a low-health player; tune `ShopsConfig.clerk.accuracy`.
- The chop shop / respray drive-in volumes are spheres in front of the garage doors (no garage geometry); the landmark buildings themselves come from `world`.
- Store interiors use ~60–120 meshes each with cached materials; no instancing. Only the active interior is visible so the cost is bounded, but the diner's checkerboard tiles are individual meshes.
- Guards do not report the player to the wanted system (jewelry/bank robberies are Agent 10 mission content); they just shoot.
- `time:dayChanged` reopens stores via `closedUntilDay > day`; `time.day` jumps by the debug clock also reopen them.

## Cut scope

- Wardrobe recolour in the safehouse (bed save point only).
- Jewelry/bank robbery flow (missions).
- Phone Feed persistence, Snapmatic gallery beyond the last shot.
- Vice Sprays / chop shop garage interiors — drive-in volumes only.
- Register cash for jewelry/bank (not robbable by the generic flow, by design).
