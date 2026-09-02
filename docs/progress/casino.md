# Agent 6 — Casino (`casino`)

## Works

- Pure logic in `src/casino/` (no Three/DOM imports, injectable `rng`), 29 vitest tests / ~200 assertions in `tests/casino/`:
  - `cards.ts`: `Card`, `parseCard('AS')`, `handValue` (soft/hard, aces demoted), `Shoe` (6 decks, reshuffles when ≤ 25 % remains, `force(codes)` pushes rigged cards to the front).
  - `Blackjack.ts`: table limits, `deal(bet)`, `hit/stand/double/split`, `canHit/canStand/canDouble/canSplit`, `resolve()` → per-hand `HandOutcome` + payouts + net. Rules: double on the first two cards only (one card then stand), one split, no re-split, split aces get one card each, 21 after a split is not a natural, dealer stands on soft 17, natural pays 3:2, push returns the bet, dealer only plays when a live hand remains.
  - `Roulette.ts`: American wheel (`WHEEL_ORDER`, `pocketColor`, 37 = `00`), bet union (straight/split/street/corner/red/black/odd/even/low/high/dozen/column), `isValidBet` (layout adjacency incl. 0/00 splits), `betCovers`, `settle(bets, outcome)` → per-bet returns + net, `spin(rng)`, `pocketAngle` for the 3D wheel, `describeBet`.
  - `Slots.ts`: 3 reels with per-reel weights, bets 10/50/100, paytable (2 cherries 2×, 3 cherries 5×, bars 10×, sevens 50×, gators 100×, diamonds 500× jackpot), `spin(bet, rng)`, `evaluate(reels, bet)`, `reelStrip`, `expectedReturn()` ≈ 92 % RTP (tested analytically and empirically).
  - `LuckyWheel.ts`: 20 weighted segments ($500–$10,000 cash, 1,000/2,500 chips, 3 mystery, 1 sports car), `canSpin(day)`, `spin(day, forcedIndex?)`, `rollMystery(rng)`.
- `CasinoSystem` (`src/systems/casino/index.ts`, `PRIORITY.casino`, `implements ICasinoPeer`): chips balance published to `game.state.chips` (+ private `casino:chipsChanged`), `buyChips` via `economy.trySpend(n, 'casino chips')`, `cashOut` via `economy.add(n, 'casino cashout')`, `play(game)` opens the table UI when inside. Registers the interior immediately if `game.interiors` exists, else on `game:ready`. Translates `interior:entered/exited {id:'casino'}` into `casino:entered/exited`, starts/stops `casino.ambient.loop`. `settleRound` emits `casino:won/lost {game, amount}` on every settled round, tracks `stats.casinoWon/casinoLost`, notifies + `casino.win.big` on wins ≥ 1,000. `serialize/deserialize`: `{chips, lastWheelDay, lockedUntilDay}`.
- Interior (`src/world/interiors/CasinoInterior.ts` + `src/systems/casino/props.ts` + `textures.ts`): 40×30 m room built in local coordinates under `ctx.root`; canvas carpet, plum walls with gold skirting/cornice, dark ceiling with 3 emissive strips, doorway accent; cashier cage with bars and a box dealer, two blackjack tables (felt canvas with printed limits, chip rails, dealer placeholders, illuminated limit signs $50–$2,000 / $500–$10,000), roulette table (layout canvas + spinning wheel cylinder with a ball that eases onto the winning pocket), 6 slot machines with emissive canvas screens (blur while spinning, show the reels on stop), Lucky Wheel (vertical disc, canvas segments, pointer, tick callback per segment), bar counter with bottles, 5 point lights. Static cuboid colliders for floor (oversized safety slab), walls and every piece of furniture via `ctx.physics.addStaticCuboid(ctx.toWorld(...))`. Per-frame `update` reuses one prompt object and a scratch `Vector3` (no allocations); nearest station within its radius pushes `[E] Cashier` / `[E] Blackjack ($50–$2,000)` / `[E] Blackjack ($500–$10,000)` / `[E] Roulette` / `[E] Slots` / `[E] Lucky Wheel` and `KeyE` opens the panel when no UI is open.
- UI (`src/ui/casino/`, appended to `#ui`, `input.pushBlocker('casino')` + `exitPointerLock`, Esc/Close, own CSS with `styles.css` variables, `ui.open/close/select/error` sounds):
  - `CashierPanel`: presets 500/1,000/5,000/10,000 + number input for buy and cash out; live cash/chips readout.
  - `BlackjackPanel`: bet input + presets clamped to limits, Deal, Hit/Stand/Double/Split/New hand enabled by the rules, HTML cards animated in one at a time, hole card face down until the player is done, "Dealer shows 10", outcome text with chip delta; `casino.card.deal/flip/shuffle`, `casino.chip.place`, `casino.blackjack/bust/push/win/lose`. Closing mid-round keeps the hand on the table.
  - `RoulettePanel`: full clickable layout (0/00, 1–36 with correct colours, columns, dozens, even-money), chip denominations 5/25/100/500, inside-bet modes straight/split/street/corner (split = two adjacent clicks), stacked chip markers, bet list, Undo/Clear (refund), Spin → 5 s: 3D wheel + ball animate in the interior, rolling number display in the UI slows and lands on the outcome, then settle; `casino.roulette.spin.loop/ball.loop/drop/win`, `casino.chip.*`. Unspun chips are refunded on close.
  - `SlotsPanel`: bet 10/50/100, "Pull lever", three reel columns cycling and stopping at 0.8/1.3/1.8 s with `casino.reel.stop`, `casino.reel.spin.loop`, `casino.slot.lever/coin/win`, `casino.jackpot`; the nearest 3D machine's screen mirrors the spin/result. Paytable shown.
  - `WheelPanel`: one spin per `game.time.day` ("Come back tomorrow" + `ui.error` otherwise), 6 s eased spin mirrored by the 3D disc with `casino.wheel.tick` per segment, `casino.wheel.stop/prize`; prizes: cash → `economy.add`, chips → chips, mystery → `weapons.give('smg', 60)` / `weapons.addAmmo('pistol', 100)` / `player.addArmor(100)` (armor when weapons absent), vehicle → `vehicles.spawn('sports', door + 6 m along the door heading, heading, {parked:true})` + "A sports car is waiting outside" (pays $10,000 when vehicles are absent).
- Guards (`src/systems/casino/guards.ts`): on `casino:entered` spawns two `guard` peds (if `game.peds` exists) flanking the entrance, despawned on exit. `weapon:fired` by the player inside → guards get `weaponId 'pistol'` and a `GuardBrain` that faces the player and calls `weapons.npcFire(ped, 'pistol', eye, dir, 0.5)` every 0.8 s; `lockedUntilDay = day + 1`, `casino:locked`, `casino.guard.alert`, notification; on exit `crime:committed {type:'rob_store', position: door, actor:'player', loud:false}` once. While banned, `interior:entered` triggers `interiors.exit()` on the next frame (after the fade finishes) with "Leonida Palms: you are banned until tomorrow" + `door.locked`. Guards reset when the ban day passes.
- Rigging (`src/systems/casino/rig.ts`): one-shot deterministic outcomes for the next round of each game, consumed by the panels.
- Autotest `src/debug/autotests/casino.ts` (not run this session — browser automation was forbidden): enter → `state.interior === 'casino'`; buy 1,000 → cash −1,000, chips 1,000; rigged natural (AS/KH vs 9/7) at bet 100 → chips 1,150; rigged roulette 17 with a 10 straight → +350; rigged diamond×3 at bet 10 → +5,000; rigged wheel cash segment pays once and a second spin the same day is refused; `casino:lost` emitted on a losing spin; cash out + exit; no console errors.
- `npm run typecheck`: 0 errors. `npx vitest run tests/casino`: 4 files, 29 tests pass.

## Cut or deferred

- Roulette UI shows a rolling number display rather than a drawn ball track; the ball animation lives on the 3D wheel in the interior.
- Slot reels are HTML glyphs (●● / BAR / 7 / ◆ / GATOR), not drawn symbol art; the 3D screens use the same glyph set.
- No insurance, surrender or re-split in blackjack (by design brief). One player seat.
- Guards do not chase; they hold their posts and shoot from there. No cops are called beyond the `crime:committed` event.
- The bar counter is decorative (no drinks purchase).
- `casino.enter` teleports via `player.teleport` then `interiors.enter`; it does not simulate walking through the door marker.

## Known bugs

- `src/systems/casino/index.ts` is 408 lines (slightly over the ~400 guideline; debug actions are the remainder).
- If the player closes a blackjack panel during the deal animation and reopens it, the cards appear without animation (state is correct).
- The Lucky Wheel `canSpin` compares against the last spin day only, so a save from day N loaded on day N still refuses (intended), but `deserialize` before `time.deserialize` in the same frame is harmless since checks happen on click.
- The kick-out while banned waits for `interiors.framework.transitioning` to clear, so the player sees one frame of the interior before the fade back out.

## How to test

- Isolated run: `?systems=world,player,camera,economy,interiors,casino` (add `peds,weapons,vehicles,ui,audio` for guards, mystery prizes, the car and notifications).
- Autotest: `&autotest=casino` (`npm run smoke -- --systems=world,player,camera,economy,interiors,casino --name=casino --autotest=casino`).
- Keys: walk to the casino landmark door and press `E`; inside, `E` at a station opens its panel; `Esc` or Close closes; `E` near the entrance exits.
- Debug actions: `casino.enter`, `casino.buy:<n>`, `casino.cashout:<n>`, `casino.chips`, `casino.play:<blackjack|roulette|slots|wheel>`, `casino.rig:<game>:<json>` (blackjack `["AS","9D","KH","7C"]` dealt player, dealer, player, dealer…; roulette pocket `0..37` (37 = 00); slots `["diamond","diamond","diamond"]`; wheel segment `0..19`), `casino.close`, `casino.lock[:days]`, `casino.unlock`.
- Example URL: `?systems=world,player,camera,economy,interiors,casino&actions=casino.enter;casino.buy:5000;casino.rig:slots:["diamond","diamond","diamond"];casino.play:slots`.

## Integration requests

- None required. Everything compiles against the current `Events.ts` / `peers.ts` / `Interiors.ts` contracts. Optional nicety (not needed): a `GameState.stats` key list could document `casinoWon` / `casinoLost`, which this system writes.

## Public API

- `game.casino` (`ICasinoPeer`): `chips: number`, `buyChips(amount): boolean`, `cashOut(amount): boolean`, `play(game: 'blackjack'|'roulette'|'slots'|'wheel'): void`.
- Extra on `CasinoSystem` (for missions/tests): `isLocked`, `isInside`, `panel` (open `CasinoPanel` or null), `enter(): Promise<boolean>`, `takeChips(n): boolean`, `giveChips(n)`, `settleRound(game, net)`, `applyWheelPrize(segment): string`, `notify(text, kind)`, `closePanel()`, `rig: CasinoRig`, `rng`, `tables.low/high: Blackjack`, `wheel: LuckyWheel`, `stage: CasinoStage` (`spinRoulette`, `spinWheel`, `setSlotScreen`, `nearestSlot`, `built`).
- Panels (`src/ui/casino/`): `BlackjackPanel.setBet/deal/hit/stand/double/split/newHand`, `RoulettePanel.selectChip/setMode/placeStraight/placeInside/placeOutside/undo/clear/spin` + `bets`, `SlotsPanel.setBet/spin` + `lastResult`, `WheelPanel.spin(): boolean` + `lastSpin`. DOM hooks: `[data-panel]`, `[data-action]`, `[data-cell="p17"]`, `[data-field="outcome|payout|prize|bet"]`, `.sl-reel[data-symbol]`.
- Events: emits `casino:entered/exited {id}`, `casino:won/lost {game, amount}`, `crime:committed` (rob_store) after a weapons incident; private `casino:chipsChanged {chips, delta}`, `casino:locked {untilDay}`, `casino:guardsAlerted {count}`.
- State: `game.state.chips`, `game.state.stats.casinoWon/casinoLost`.
- Pure modules: `src/casino/{cards,Blackjack,Roulette,Slots,LuckyWheel}.ts` — usable by other systems (e.g. a mission that rigs a hand) without any Three/DOM dependency.
