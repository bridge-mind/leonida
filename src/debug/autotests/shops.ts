import type { Game } from '../../core/Game';
import { ARMOR_AMOUNT, FOOD, WEAPON_PRICES } from '../../data/prices';
import type { ShopSystem } from '../../systems/shops';
import { assert, Suite, waitFor, waitFrames } from './util';

/**
 * Shops acceptance (docs/TEAM-PLAN.md Agent 9): Bullet Barn purchases, food, the
 * safehouse save point, the phone and the closed-store gate.
 * Run: npm run smoke -- --systems=world,player,camera,economy,interiors,weapons,peds,pickups,ui,navigation,shops,chopshop,robbery --name=shops --autotest=shops
 */
export default async function shops(game: Game): Promise<void> {
  const suite = new Suite('shops', game);
  const system = game.systems.get<ShopSystem>('shops');
  assert(system, 'shops system missing');
  const economy = game.economy;
  assert(economy, 'economy system missing');
  const cash = (): number => game.debug.run('economy.get') as number;
  const status = (): Record<string, unknown> => game.debug.run('shops.status') as Record<string, unknown>;

  await suite.check('interiors-registered', async () => {
    await waitFor(() => game.interiors?.has('gunstore') ?? false, 5000, 'gunstore interior registration');
    for (const id of ['store_habana', 'store_gellhorn', 'store_downtown', 'store_beach', 'gas_station', 'waffle_hut', 'safehouse', 'jewelry', 'bank']) {
      assert(game.interiors?.has(id), `interior ${id} not registered`);
    }
    assert(game.systems.has('chopshop'), 'chopshop system missing (src/systems/chopshop must be in the --systems list)');
  });

  await suite.check('enter-gunstore', async () => {
    await game.debug.run('interiors.enter', 'gunstore');
    await waitFor(() => game.state.interior === 'gunstore', 6000, 'interior:entered gunstore');
    await waitFrames(game, 3);
    assert(status().current === 'gunstore', 'shop system did not register the visit');
    if (game.systems.has('peds')) assert(status().clerk !== null, 'no clerk spawned in the Bullet Barn');
  });

  await suite.check('buy-rifle', async () => {
    const weapons = game.weapons;
    assert(weapons, 'weapons system missing');
    game.debug.run('economy.set', 10000);
    const before = cash();
    const result = game.debug.run('shops.buy', 'rifle') as { ok: boolean; message: string };
    assert(result.ok, `purchase refused: ${result.message}`);
    assert(cash() === before - WEAPON_PRICES.rifle.price, `expected cash ${before - 3000}, got ${cash()}`);
    assert(weapons.has('rifle'), 'rifle not in inventory after purchase');
  });

  await suite.check('buy-ammo-when-owned', async () => {
    const weapons = game.weapons;
    assert(weapons, 'weapons system missing');
    const ammoBefore = weapons.inventory().rifle ?? 0;
    const before = cash();
    const result = game.debug.run('shops.buy', 'rifle') as { ok: boolean; message: string };
    assert(result.ok, `ammo purchase refused: ${result.message}`);
    assert(cash() === before - WEAPON_PRICES.rifle.ammo, `ammo should cost ${WEAPON_PRICES.rifle.ammo}`);
    assert((weapons.inventory().rifle ?? 0) > ammoBefore, 'ammo count did not increase');
  });

  await suite.check('deny-when-broke', async () => {
    game.debug.run('economy.set', 10);
    const result = game.debug.run('shops.buy', 'sniper') as { ok: boolean };
    assert(!result.ok, 'sniper purchase should be refused with $10');
    assert(cash() === 10, 'cash changed on a refused purchase');
    game.debug.run('economy.set', 5000);
  });

  await suite.check('buy-armor', async () => {
    game.debug.run('player.armor', 0);
    const before = cash();
    const result = game.debug.run('shops.buy', 'armor') as { ok: boolean; message: string };
    assert(result.ok, `armour refused: ${result.message}`);
    assert(game.player.armor === ARMOR_AMOUNT, `armour should be ${ARMOR_AMOUNT}, got ${game.player.armor}`);
    assert(cash() === before - 500, 'armour should cost $500');
  });

  await suite.check('gunstore-ui-open-close', async () => {
    game.debug.run('shops.open', 'gunstore');
    await waitFrames(game, 2);
    assert(status().gunStoreOpen === true, 'gun store panel did not open');
    assert(game.input.isBlocked, 'gun store panel should block gameplay input');
    assert(document.querySelector('.shop-panel') !== null, 'gun store DOM missing');
    game.debug.run('shops.closeUi');
    await waitFrames(game, 2);
    assert(status().gunStoreOpen === false, 'gun store panel did not close');
    assert(!game.input.isBlocked, 'input blocker left behind');
  });

  await suite.check('food-heals', async () => {
    await game.debug.run('interiors.exit');
    await waitFor(() => game.state.interior === null, 6000, 'interior:exited gunstore');
    await game.debug.run('interiors.enter', 'waffle_hut');
    await waitFor(() => game.state.interior === 'waffle_hut', 6000, 'interior:entered waffle_hut');
    await waitFrames(game, 2);
    game.debug.run('player.damage', 60);
    const hp = game.player.health;
    const before = cash();
    const result = game.debug.run('shops.food') as { ok: boolean; message: string };
    assert(result.ok, `food refused: ${result.message}`);
    assert(game.player.health >= Math.min(100, hp + FOOD.waffle.heal) - 0.01, `waffle should heal ${FOOD.waffle.heal}`);
    assert(cash() === before - FOOD.waffle.price, 'waffle should cost $15');
  });

  await suite.check('safehouse-save-point', async () => {
    await game.debug.run('interiors.exit');
    await waitFor(() => game.state.interior === null, 6000, 'interior:exited waffle_hut');
    await game.debug.run('interiors.enter', 'safehouse');
    await waitFor(() => game.state.interior === 'safehouse', 6000, 'interior:entered safehouse');
    const rt = system.runtime('safehouse');
    assert(rt?.layout.bed, 'safehouse has no bed point');
    assert(rt.clerk === null, 'safehouse should be unstaffed');
  });

  await suite.check('store-closed-gate', async () => {
    await game.debug.run('interiors.exit');
    await waitFor(() => game.state.interior === null, 6000, 'interior:exited safehouse');
    game.debug.run('shops.close', 'store_beach');
    assert(system.isStoreClosed('store_beach'), 'store_beach should be closed');
    assert(!system.isStoreClosed('store_habana'), 'store_habana should still be open');
    game.debug.run('shops.reopen', 'store_beach');
    assert(!system.isStoreClosed('store_beach'), 'store_beach should reopen');
  });

  await suite.check('phone-open-snap', async () => {
    game.debug.run('phone.open');
    await waitFrames(game, 2);
    assert(system.phone.isOpen, 'phone did not open');
    assert(game.input.isBlocked, 'phone should block gameplay input');
    const url = (await game.debug.run('phone.snap')) as string;
    assert(typeof url === 'string' && url.startsWith('data:image/png'), `snapmatic returned ${String(url).slice(0, 30)}`);
    game.debug.run('phone.app', 'feed');
    const feed = game.debug.run('phone.feed') as string[];
    assert(feed.length > 0, 'feed should have at least the welcome post');
    game.debug.run('phone.close');
    await waitFrames(game, 2);
    assert(!system.phone.isOpen, 'phone did not close');
    assert(!game.input.isBlocked, 'phone left an input blocker behind');
  });

  await suite.check('hospital-fee-once', async () => {
    game.debug.run('economy.set', 10000);
    const before = cash();
    game.debug.run('player.kill');
    await waitFor(() => game.player.isAlive && game.player.health > 0 && cash() < before, 15000, 'respawn with hospital bill');
    await waitFrames(game, 3);
    const paid = before - cash();
    // Player system bills 10% or economy bills 5% (min $100) — never both.
    assert(paid === 1000 || paid === 500, `hospital fee should be charged once (5% or 10%), got $${paid}`);
  });

  suite.finish();
}
