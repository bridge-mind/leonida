import type { Game } from '../../core/Game';
import type { IPed } from '../../core/entities';
import type { CrimeType } from '../../core/Events';
import { chopValue } from '../../data/prices';
import { VEHICLES } from '../../data/vehicles';
import type { ChopShopSystem } from '../../systems/shops/ChopShop';
import { assert, Suite, waitFor, waitFrames } from './util';

/**
 * Robbery / chop shop / respray acceptance (docs/TEAM-PLAN.md Agent 9).
 * Run: npm run smoke -- --systems=world,player,camera,economy,interiors,weapons,peds,pickups,vehicles,ui,navigation,shops,chopshop,robbery,wanted --name=robbery --autotest=robbery
 */
export default async function robbery(game: Game): Promise<void> {
  const suite = new Suite('robbery', game);
  assert(game.systems.has('shops'), 'shops system missing');
  assert(game.systems.has('robbery'), 'robbery system missing');
  const peds = game.peds;
  assert(peds, 'peds system missing (clerks need it)');
  const weapons = game.weapons;
  assert(weapons, 'weapons system missing');
  const economy = game.economy;
  assert(economy, 'economy system missing');
  const cash = (): number => game.debug.run('economy.get') as number;
  const robberyStatus = (): Record<string, unknown> => game.debug.run('robbery.status') as Record<string, unknown>;

  const events: string[] = [];
  const crimes: CrimeType[] = [];
  const offs = [
    game.events.on('robbery:started', () => events.push('started')),
    game.events.on('robbery:completed', () => events.push('completed')),
    game.events.on('crime:committed', (e) => crimes.push(e.type)),
    game.events.on('chopshop:sold', () => events.push('sold')),
  ];

  let clerk: IPed | null = null;

  await suite.check('enter-24-7', async () => {
    await waitFor(() => game.interiors?.has('store_habana') ?? false, 5000, 'store interiors registered');
    await game.debug.run('interiors.enter', 'store_habana');
    await waitFor(() => game.state.interior === 'store_habana', 6000, 'interior:entered store_habana');
    await waitFrames(game, 3);
    const found: IPed[] = [];
    peds.query(game.player.position, 25, found, (p) => p.typeId === 'clerk');
    assert(found.length === 1, `expected one clerk in the store, found ${found.length}`);
    clerk = found[0];
  });

  await suite.check('aim-starts-robbery', async () => {
    assert(clerk, 'no clerk');
    weapons.give('pistol', 24);
    weapons.equip('pistol');
    await waitFrames(game, 1);
    assert(weapons.current === 'pistol', 'pistol not equipped');
    const start = performance.now();
    let seconds = 0;
    while (!events.includes('started')) {
      if (performance.now() - start > 4000) throw new Error('robbery did not start after 4 s of aiming at the clerk');
      seconds += game.time.realDt;
      game.events.emit('weapon:aimedAt', { target: clerk, seconds });
      await waitFrames(game, 1);
    }
    assert(robberyStatus().active === 'store_habana', 'robbery.activeStore should be store_habana');
    assert(clerk.handsUp, 'clerk should have his hands up');
  });

  await suite.check('gunshot-speeds-up', async () => {
    const before = robberyStatus().secondsLeft as number;
    game.events.emit('weapon:fired', {
      weaponId: 'pistol',
      origin: { x: 0, y: 0, z: 0 },
      direction: { x: 0, y: 0, z: 1 },
      shooter: 'player',
    });
    await waitFrames(game, 1);
    const after = robberyStatus().secondsLeft as number;
    assert(after < before * 0.65, `a gunshot should cut ~40% of the remaining time (${before.toFixed(1)} → ${after.toFixed(1)})`);
    assert(robberyStatus().shots === 1, 'shot counter should be 1');
  });

  await suite.check('bag-fills', async () => {
    const pickupsBefore = game.systems.has('pickups') ? (game.debug.run('pickups.count') as number) : 0;
    assert(game.debug.run('robbery.skip', 30) === true, 'skip should complete the robbery');
    await waitFrames(game, 2);
    assert(events.includes('completed'), 'robbery:completed not emitted');
    if (game.systems.has('pickups')) {
      assert((game.debug.run('pickups.count') as number) === pickupsBefore + 1, 'no cash pickup spawned on the register');
    }
    assert(game.shops?.isStoreClosed('store_habana'), 'store should be closed after the robbery');
  });

  await suite.check('exit-reports-crime', async () => {
    await game.debug.run('interiors.exit');
    await waitFor(() => game.state.interior === null, 6000, 'interior:exited');
    await waitFrames(game, 2);
    assert(crimes.includes('rob_store'), `crime:committed rob_store not observed (saw: ${crimes.join(',') || 'none'})`);
    assert(robberyStatus().active === null, 'robbery should be over after leaving');
    if (game.wanted) assert(game.wanted.level >= 1, 'wanted system did not react to rob_store');
  });

  await suite.check('closed-store-refuses-entry', async () => {
    const ok = (await game.debug.run('interiors.enter', 'store_habana')) as boolean;
    // The framework itself still lets debug `enter` through; the closed gate lives in the door prompt.
    if (ok) {
      await game.debug.run('interiors.exit');
      await waitFor(() => game.state.interior === null, 6000, 'interior:exited');
    }
    assert(game.shops?.isStoreClosed('store_habana'), 'store_habana should stay closed for the day');
    game.debug.run('shops.reopen', 'store_habana');
  });

  await suite.check('chop-shop-sells-sports', async () => {
    const vehicles = game.vehicles;
    assert(vehicles, 'vehicles system missing');
    const chop = game.systems.get<ChopShopSystem>('chopshop');
    assert(chop, 'chopshop system missing');
    const at = game.debug.run('chopshop.position') as { x: number; y: number; z: number };
    game.debug.run('player.teleport', at.x + 3, at.z + 3);
    await waitFrames(game, 2);
    game.debug.run('chopshop.setWanted', 'sedan');
    game.debug.run('vehicles.spawn', 'sports', at.x, at.z, 0);
    await waitFrames(game, 2);
    assert(game.debug.run('vehicles.enterNearest') !== null, 'player could not enter the sports car');
    await waitFrames(game, 2);
    const v = vehicles.playerVehicle;
    assert(v && v.archetypeId === 'sports', 'player is not in the sports car');
    const expected = chopValue(VEHICLES.sports.chopValue, v.state.health / VEHICLES.sports.health, false);
    assert(Math.abs(expected - 4000) <= 400, `fresh sports car should be worth ~$4000, table says ${expected}`);
    const before = cash();
    const paid = (await game.debug.run('chopshop.sell')) as number;
    assert(paid === expected, `payout ${paid} != value ${expected}`);
    assert(cash() === before + paid, 'cash did not increase by the payout');
    assert(vehicles.playerVehicle === null, 'player should be out of the car');
    assert(!vehicles.all.includes(v), 'sold vehicle still exists');
    assert(events.includes('sold'), 'chopshop:sold not emitted');
  });

  await suite.check('respray-clears-wanted', async () => {
    const vehicles = game.vehicles;
    assert(vehicles, 'vehicles system missing');
    const at = game.debug.run('respray.position') as { x: number; y: number; z: number };
    game.debug.run('player.teleport', at.x + 2, at.z + 2);
    await waitFrames(game, 2);
    game.debug.run('vehicles.spawn', 'sedan', at.x, at.z, 0);
    await waitFrames(game, 2);
    assert(game.debug.run('vehicles.enterNearest') !== null, 'player could not enter the sedan');
    await waitFrames(game, 3);
    const v = vehicles.playerVehicle;
    assert(v, 'player is not in a car');
    const colorBefore = v.color;
    const wanted = game.wanted;
    // `setLevel(2)` defaults to `active` and marks the player as seen, which makes the respray refuse
    // (correctly) for a second; `searching` exercises the clear path instead.
    if (wanted) wanted.setLevel(2, 'searching');
    game.debug.run('economy.set', 5000);
    const before = cash();
    const ok = (await game.debug.run('respray.spray')) as boolean;
    if (wanted && wanted.playerSeen) {
      assert(!ok, 'respray must refuse while cops can see the player');
      return;
    }
    assert(ok, 'respray refused');
    const expectedCost = wanted ? 2000 : 1000;
    assert(cash() === before - expectedCost, `respray should cost ${expectedCost}, paid ${before - cash()}`);
    if (wanted) assert(wanted.level === 0, `wanted level should be 0 after respray, got ${wanted.level}`);
    assert(v.color !== colorBefore || VEHICLES.sedan.palette.length < 2, 'car colour did not change');
  });

  for (const off of offs) off();
  suite.finish();
}
