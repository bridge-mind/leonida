import type { Game } from '../../core/Game';
import type { PedSystem } from '../../systems/peds';
import type { PickupSystem } from '../../systems/pickups';
import { assert, horizontalDistance, Suite, wait, waitFor, waitFrames } from './util';

/**
 * Pedestrian + pickup acceptance (TEAM-PLAN §5 Agent 3).
 * Run: npm run smoke -- --systems=world,player,camera,economy,peds,pickups --name=peds --autotest=peds
 */
export default async function peds(game: Game): Promise<void> {
  const suite = new Suite('peds', game);
  const sys = game.systems.get<PedSystem>('peds');
  assert(sys, 'peds system missing');
  const pickups = game.systems.get<PickupSystem>('pickups');
  const player = game.player;
  const graph = game.world.roadGraph;
  const run = (path: string, ...args: unknown[]): unknown => game.debug.run(path, ...args);
  const stateOf = (id: number): string | null => sys.byId(id)?.state ?? null;

  // Downtown: dense sidewalks, daytime density 1.0.
  game.time.setHour(12);
  player.teleport({ x: -20, y: 0, z: 120 }, Math.PI);
  game.camera.setOrbit(player.heading + Math.PI, 0.25);
  await waitFrames(game, 5);

  await suite.check('crowd-spawns-without-errors', async () => {
    await wait(8000);
    const count = sys.live.length;
    const stats = run('peds.stats') as Record<string, number>;
    assert(count >= 40, `only ${count} peds live after 8 s (target ${stats.target})`);
    assert(game.debug.errorCount === 0, `${game.debug.errorCount} console errors: ${game.debug.errors.join(' | ')}`);
  });

  await suite.check('aim-cower-rob-pickup-cash', async () => {
    const id = run('peds.spawnHere', 'business') as number;
    await waitFrames(game, 2);
    const aimed = run('peds.aimAt', id) as number | null;
    assert(aimed === id, `aimAt picked ped ${aimed}, expected the spawned ${id}`);
    await waitFor(() => stateOf(id) === 'cower', 1000, `ped ${id} to cower (state ${stateOf(id)})`);
    let cowered = false;
    let robbed = false;
    const offA = game.events.on('ped:cowered', (e) => {
      if (e.ped.id === id) cowered = true;
    });
    const offB = game.events.on('crime:committed', (e) => {
      if (e.type === 'rob_ped') robbed = true;
    });
    assert(pickups, 'pickups system missing');
    const before = pickups.live.length;
    await waitFor(() => pickups.live.length > before || robbed, 3000, 'cash pickup after 1.5 s of aiming');
    offA();
    offB();
    assert(robbed, 'crime:committed rob_ped not emitted');
    assert(pickups.live.some((p) => p.pickupKind === 'cash'), 'no cash pickup live');
    // The ped's state was set by the robbery flow; cowered event may have fired before the listener.
    assert(cowered || stateOf(id) === 'flee' || stateOf(id) === 'wander', `ped ${id} did not flee after the robbery (state ${stateOf(id)})`);

    const economy = game.economy;
    if (!economy) {
      console.log('AUTOTEST SKIP peds/pickup-adds-cash (economy not loaded)');
      return;
    }
    const cash = economy.cash;
    game.camera.setOrbit(player.heading + Math.PI, 0.25);
    await waitFrames(game, 3);
    game.input.hold('KeyW');
    try {
      await waitFor(() => economy.cash > cash, 6000, 'walking into the cash pickup to add cash');
    } finally {
      game.input.release('KeyW');
    }
    assert(economy.cash > cash, 'cash did not increase');
  });

  await suite.check('gunshot-flee-and-report', async () => {
    // A ring of extra witnesses makes the 10% call-police roll statistically certain.
    const p = player.position;
    for (let i = 0; i < 20; i++) {
      const a = (i / 20) * Math.PI * 2;
      run('peds.spawn', 'tourist', p.x + Math.cos(a) * 15, p.z + Math.sin(a) * 15);
    }
    await waitFrames(game, 3);
    let reported = false;
    const off = game.events.on('ped:reportedCrime', () => {
      reported = true;
    });
    run('peds.gunshot');
    await wait(3000);
    let near = 0;
    let reacting = 0;
    for (const ped of sys.live) {
      if (!ped.isAlive || horizontalDistance(ped.position, p) > 40) continue;
      near++;
      if (ped.state === 'flee' || ped.state === 'callPolice') reacting++;
    }
    assert(near > 0, 'no peds within 40 m');
    assert(reacting / near >= 0.5, `only ${reacting}/${near} peds within 40 m are fleeing or calling police`);
    await waitFor(() => reported, 3500, 'ped:reportedCrime within 6 s of the shot');
    off();
  });

  await suite.check('vehicle-damage-ragdoll-despawn', async () => {
    const id = run('peds.spawnHere', 'worker') as number;
    await waitFrames(game, 2);
    let died = false;
    const off = game.events.on('ped:died', (e) => {
      if (e.ped.id === id) died = true;
    });
    run('peds.damage', id, 90, 'vehicle');
    run('peds.ragdoll', id);
    assert(stateOf(id) === 'knockedDown', `survivor should be knocked down while ragdolling (state ${stateOf(id)})`);
    run('peds.damage', id, 90, 'vehicle');
    off();
    assert(died, 'ped:died not emitted');
    assert(stateOf(id) === 'dead', `state ${stateOf(id)} after lethal vehicle damage`);
    run('peds.fastForward', 22);
    await waitFor(() => sys.byId(id) === undefined, 3000, 'ragdoll despawn after the freeze + despawn timers');
  });

  await suite.check('wanderers-stay-on-sidewalks', async () => {
    await wait(1500);
    let sampled = 0;
    let onSidewalk = 0;
    for (const ped of sys.live) {
      if (ped.state !== 'wander' || sampled >= 50) continue;
      sampled++;
      const hit = graph.nearestSidewalkPoint(ped.position);
      if (hit && horizontalDistance(hit.point, ped.position) <= 3) onSidewalk++;
    }
    assert(sampled >= 10, `only ${sampled} wandering peds to sample`);
    assert(onSidewalk / sampled >= 0.9, `${onSidewalk}/${sampled} wanderers within 3 m of a sidewalk`);
  });

  await suite.check('no-errors-after-run', () => {
    assert(game.debug.errorCount === 0, `${game.debug.errorCount} console errors: ${game.debug.errors.join(' | ')}`);
  });

  suite.finish();
}
