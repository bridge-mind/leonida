import type { IVehicle } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { RoadNode } from '../../world/RoadGraph';
import { assert, horizontalDistance, Suite, wait, waitFrames } from './util';

/**
 * Traffic acceptance: density converges, nobody stalls on a green light, cars stop on red.
 * Run: ?systems=world,player,camera,vehicles,traffic&autotest=traffic
 */
export default async function traffic(game: Game): Promise<void> {
  const suite = new Suite('traffic', game);
  const graph = game.world.roadGraph;
  const run = (path: string, ...args: unknown[]): unknown => game.debug.run(path, ...args);

  await suite.check('systems-loaded', () => {
    assert(game.ready, 'game not ready');
    for (const id of ['world', 'player', 'camera', 'vehicles', 'traffic']) assert(game.systems.has(id), `system ${id} missing`);
    assert(game.traffic && game.vehicles, 'traffic/vehicles peers missing');
  });
  const trafficPeer = game.traffic;
  const vehiclesPeer = game.vehicles;
  assert(trafficPeer && vehiclesPeer, 'peers missing');

  // Stand in the densest cluster of lit intersections so all three sampled lights get traffic.
  const lights = graph.nodes.filter((n) => n.hasLight);
  assert(lights.length >= 3, `only ${lights.length} light nodes`);
  const sampled = closestTriple(lights);
  const center = {
    x: (sampled[0].position.x + sampled[1].position.x + sampled[2].position.x) / 3,
    z: (sampled[0].position.z + sampled[1].position.z + sampled[2].position.z) / 3,
  };
  game.player.teleport({ x: center.x, y: 0, z: center.z }, 0);
  game.camera.setOrbit(0, 0.4);
  await waitFrames(game, 5);

  await suite.check('density-converges-15s', async () => {
    await wait(15000);
    const target = (run('traffic.count') as { target: number }).target;
    const count = trafficPeer.count;
    assert(Math.abs(count - target) <= 6, `traffic count ${count}, target ${target}`);
  });

  const scratch: IVehicle[] = [];
  const stopped = new Set<number>();
  let stalled = 0;

  await suite.check('cars-stop-on-red-at-3-lights', async () => {
    const t0 = performance.now();
    while (performance.now() - t0 < 60000 && stopped.size < 3) {
      for (let i = 0; i < 3; i++) {
        const node = sampled[i];
        if (stopped.has(node.id)) continue;
        vehiclesPeer.query(node.position, 12, scratch);
        for (const v of scratch) {
          if (v.state.speedMs > 0.5) continue;
          const lp = graph.nearestLanePoint(v.position);
          if (!lp || lp.lane.to !== node.id) continue;
          if (!trafficPeer.isGreen(node.id, lp.edge.id)) stopped.add(node.id);
        }
      }
      stalled = Math.max(stalled, run('traffic.stalled', 20) as number);
      await wait(250);
    }
    assert(stopped.size >= 3, `cars stopped on red at only ${stopped.size}/3 sampled lights (${sampled.map((n) => n.id).join(',')})`);
  });

  await suite.check('no-car-stalls-on-green', () => {
    // Budget: 20 s of idling with a green light and a clear road (spec allows 60 s; the test is shorter).
    assert(stalled === 0, `${stalled} car(s) idled > 20 s with a green light and nothing ahead`);
  });

  await suite.check('panic-clears-drivers', async () => {
    const before = trafficPeer.count;
    const bailed = run('traffic.panic') as number;
    assert(bailed >= 0, 'panic returned a negative count');
    assert(trafficPeer.count === before - bailed, 'count did not drop by the bailed number');
    await waitFrames(game, 3);
    vehiclesPeer.query(game.player.position, 60, scratch);
    for (const v of scratch) {
      if (v === vehiclesPeer.playerVehicle) continue;
      assert(v.driver === null || horizontalDistance(v.position, game.player.position) > 60, `vehicle ${v.id} still driven after panic`);
    }
  });

  await suite.check('no-console-errors', () => {
    assert(game.debug.errorCount === 0, `${game.debug.errorCount} console errors: ${game.debug.errors.join(' | ')}`);
  });

  suite.finish();
}

/** Three light nodes with the smallest bounding radius, so one spawn bubble covers them all. */
function closestTriple(nodes: RoadNode[]): [RoadNode, RoadNode, RoadNode] {
  let best: [RoadNode, RoadNode, RoadNode] = [nodes[0], nodes[1], nodes[2]];
  let bestSpread = Infinity;
  for (const a of nodes) {
    const near = nodes
      .filter((n) => n !== a)
      .sort((p, q) => horizontalDistance(p.position, a.position) - horizontalDistance(q.position, a.position))
      .slice(0, 2);
    if (near.length < 2) continue;
    const spread = horizontalDistance(near[1].position, a.position);
    if (spread < bestSpread) {
      bestSpread = spread;
      best = [a, near[0], near[1]];
    }
  }
  return best;
}
