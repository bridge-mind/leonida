import type { IVehicle } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { LanePolyline } from '../../world/RoadGraph';
import { assert, horizontalDistance, Suite, waitFor, waitFrames, wait } from './util';

interface VehicleDebugState {
  speedMs: number;
  yawRate: number;
  health: number;
  exploded: boolean;
  position: { x: number; y: number; z: number };
}

/**
 * Vehicle acceptance: spawn, enter, drive along a lane, handbrake turn, explode, exit.
 * Run: ?systems=world,player,camera,vehicles&autotest=vehicles
 */
export default async function vehicles(game: Game): Promise<void> {
  const suite = new Suite('vehicles', game);
  const player = game.player;
  const graph = game.world.roadGraph;
  const run = (path: string, ...args: unknown[]): unknown => game.debug.run(path, ...args);
  const state = (): VehicleDebugState | null => run('vehicles.state') as VehicleDebugState | null;
  const releaseAll = (): void => {
    for (const k of ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'Space']) game.input.release(k);
  };

  await suite.check('systems-loaded', () => {
    assert(game.ready, 'game not ready');
    for (const id of ['world', 'player', 'camera', 'vehicles']) assert(game.systems.has(id), `system ${id} missing`);
    assert(game.vehicles, 'game.vehicles peer missing');
  });

  // Longest flat street: plenty of straight road to accelerate on without a junction.
  const lane = longestStreetLane(graph);
  assert(lane, 'no street lane found');
  const p0 = lane.points[0];
  const p1 = lane.points[1];
  const heading = Math.atan2(p1.x - p0.x, p1.z - p0.z);
  player.teleport({ x: p0.x, y: p0.y + 0.2, z: p0.z }, heading);
  await waitFrames(game, 5);
  const before = game.vehicles ? game.vehicles.all.length : 0;
  const find = (id: number): IVehicle | undefined => game.vehicles?.all.find((v) => v.id === id);

  let vehicleId = -1;
  await suite.check('spawnHere-sedan', () => {
    vehicleId = run('vehicles.spawnHere', 'sedan') as number;
    assert(typeof vehicleId === 'number', 'spawnHere did not return an id');
    assert(game.vehicles && game.vehicles.all.length === before + 1, 'vehicle count did not grow by one');
    const v = find(vehicleId);
    assert(v, 'spawned vehicle not found via get()');
    assert(horizontalDistance(v.position, player.position) < 8, 'vehicle spawned too far from the player');
  });

  await suite.check('enterNearest-mounts-player', async () => {
    const id = run('vehicles.enterNearest') as number | null;
    assert(id === vehicleId, `enterNearest picked ${id}, expected ${vehicleId}`);
    await waitFor(() => player.isMounted, 3000, 'player to mount');
    assert(player.mountedVehicle?.id === vehicleId, 'player mounted a different vehicle');
    assert(game.state.vehicle?.id === 'sedan', 'game.state.vehicle not published');
  });

  await suite.check('drive-4s-speed-and-lane', async () => {
    game.input.hold('KeyW');
    await wait(4000);
    const s = state();
    assert(s, 'no player vehicle state while driving');
    assert(s.speedMs > 12, `speed only ${s.speedMs.toFixed(1)} m/s after 4 s`);
    const lp = graph.nearestLanePoint(s.position);
    assert(lp, 'no lane point near the vehicle');
    const off = horizontalDistance(lp.point, s.position);
    assert(off < 6, `vehicle drifted ${off.toFixed(1)} m from the lane`);
  });

  await suite.check('handbrake-turn-yaw-rate', async () => {
    game.input.hold('Space');
    game.input.hold('KeyA');
    let peak = 0;
    const t0 = performance.now();
    while (performance.now() - t0 < 1500) {
      const s = state();
      if (s) peak = Math.max(peak, Math.abs(s.yawRate));
      await wait(16);
    }
    releaseAll();
    assert(peak > 0.8, `peak yaw rate ${peak.toFixed(2)} rad/s`);
  });

  await suite.check('explodeCurrent-fires-event', async () => {
    await waitFrames(game, 10);
    let exploded = false;
    const off = game.events.on('vehicle:exploded', (e) => {
      if (e.vehicle.id === vehicleId) exploded = true;
    });
    const ok = run('vehicles.explodeCurrent') as boolean;
    off();
    assert(ok, 'explodeCurrent returned false (player not in a vehicle?)');
    assert(exploded, 'vehicle:exploded not emitted');
    const v = find(vehicleId);
    assert(v, 'exploded vehicle disappeared immediately (wreck should remain)');
    assert(v.state.health <= 0, `health ${v.state.health} after explosion`);
    assert(v.state.wrecked, 'vehicle not flagged wrecked');
    await waitFor(() => !player.isMounted, 2000, 'player to be ejected');
    assert(game.state.vehicle === null, 'game.state.vehicle still set after ejection');
  });

  await suite.check('enter-then-exit-on-foot', async () => {
    player.teleport({ x: p0.x, y: p0.y + 0.2, z: p0.z }, heading);
    await waitFrames(game, 5);
    const id = run('vehicles.spawnHere', 'suv') as number;
    const picked = run('vehicles.enterNearest') as number | null;
    assert(picked === id, `enterNearest picked ${picked}, expected the fresh suv ${id}`);
    await waitFor(() => player.isMounted, 3000, 'player to mount the suv');
    run('vehicles.exit');
    await waitFrames(game, 3);
    assert(!player.isMounted, 'player still mounted after exit');
    assert(player.isAlive, 'player died exiting a stationary car');
    const v = find(id);
    assert(v && v.driverEntity === null, 'vehicle still has a driver after exit');
    assert(horizontalDistance(player.position, v.position) < 6, 'player dismounted far from the vehicle');
  });

  await suite.check('no-console-errors', () => {
    assert(game.debug.errorCount === 0, `${game.debug.errorCount} console errors: ${game.debug.errors.join(' | ')}`);
  });

  releaseAll();
  suite.finish();
}

function longestStreetLane(graph: Game['world']['roadGraph']): LanePolyline | null {
  let best: LanePolyline | null = null;
  let bestLen = 0;
  for (const edge of graph.edges) {
    if (edge.kind === 'dirt' || edge.isHighway || edge.lanes.length === 0) continue;
    const lane = graph.lanes[edge.lanes[0]];
    let flat = true;
    for (let i = 1; i < lane.points.length && flat; i++) flat = Math.abs(lane.points[i].y - lane.points[0].y) < 0.5;
    if (flat && edge.length > bestLen) {
      bestLen = edge.length;
      best = lane;
    }
  }
  return best;
}
