import { BoxGeometry, Mesh, MeshStandardMaterial } from 'three';
import type { Game } from '../../core/Game';
import { Random } from '../../core/Random';
import { Groups } from '../../physics/Physics';
import { assert, horizontalDistance, Suite, waitFor, waitFrames, wait } from './util';

/**
 * Foundation acceptance: Phase 0 + Phase 1 criteria plus the framework contracts.
 * Run: npm run smoke -- --name=foundation-phase1 --autotest=foundation
 */
export default async function foundation(game: Game): Promise<void> {
  const suite = new Suite('foundation', game);
  const player = game.player;
  const world = game.world;
  const graph = world.roadGraph;

  await suite.check('ready-no-errors', () => {
    assert(game.ready, 'game not ready');
    assert(game.debug.errorCount === 0, `${game.debug.errorCount} console errors: ${game.debug.errors.join(' | ')}`);
    for (const id of ['world', 'player', 'camera']) assert(game.systems.has(id), `system ${id} missing`);
  });

  // Open sand strip on Vice Beach: nothing to bump into while walking north.
  player.teleport({ x: 918, y: 0, z: 120 }, Math.PI);
  game.camera.setOrbit(0, 0.2);
  await waitFrames(game, 5);

  await suite.check('walk-20m', async () => {
    const start = player.position.clone();
    game.input.hold('KeyW');
    await waitFor(() => horizontalDistance(player.position, start) >= 20, 9000, 'player to walk 20 m');
    game.input.release('KeyW');
    const moved = horizontalDistance(player.position, start);
    assert(moved >= 20, `only moved ${moved.toFixed(1)} m`);
    assert(player.position.z < start.z - 15, `walked the wrong way (dz ${(player.position.z - start.z).toFixed(1)})`);
  });

  await suite.check('jump-raises-player', async () => {
    await waitFrames(game, 10);
    const groundY = player.position.y;
    let peak = groundY;
    game.input.press('Space', 120);
    const t0 = performance.now();
    while (performance.now() - t0 < 1200) {
      peak = Math.max(peak, player.position.y);
      await wait(16);
    }
    assert(peak - groundY > 0.6, `jump peak only ${(peak - groundY).toFixed(2)} m`);
    await waitFor(() => player.entity.grounded, 3000, 'landing');
  });

  await suite.check('astar-20-random-pairs', () => {
    const rng = new Random(1234);
    assert(graph.isConnected(), 'road graph is not a single connected component');
    for (let i = 0; i < 20; i++) {
      const a = rng.int(0, graph.nodes.length - 1);
      const b = rng.int(0, graph.nodes.length - 1);
      const path = graph.astar(a, b);
      assert(path && path[0] === a && path[path.length - 1] === b, `no path between nodes ${a} and ${b}`);
    }
  });

  await suite.check('landmark-doors-near-road-nodes', () => {
    const far: string[] = [];
    for (const l of world.landmarks.list) {
      const node = graph.nearestNode(l.door);
      const d = horizontalDistance(node.position, l.door);
      if (d > 15) far.push(`${l.id} (${d.toFixed(1)} m)`);
    }
    assert(far.length === 0, `doors farther than 15 m from a node: ${far.join(', ')}`);
  });

  await suite.check('static-draw-calls-under-400', async () => {
    // Sample from a dense viewpoint: downtown looking at the skyline.
    player.teleport({ x: -20, y: 0, z: 120 }, Math.PI);
    game.camera.setOrbit(0, 0.1);
    await waitFrames(game, 5);
    const calls = game.perf.drawCalls;
    assert(calls > 0 && calls < 400, `draw calls ${calls}`);
  });

  await suite.check('dusk-turns-lamps-on', async () => {
    game.time.setHour(12);
    await waitFrames(game, 2);
    assert(world.night01 < 0.05, `lamps on at noon (${world.night01.toFixed(2)})`);
    game.time.setHour(20);
    await waitFrames(game, 2);
    assert(world.night01 > 0.95, `lamps off at 20:00 (${world.night01.toFixed(2)})`);
    assert(world.city.materials.walls.emissiveIntensity > 1, 'window emissive not lit at night');
    game.time.setHour(18.5);
  });

  await suite.check('water-query', () => {
    assert(world.isInWater({ x: 425, y: -0.3, z: 200 }), 'bay channel should be water');
    assert(!world.isInWater({ x: 425, y: 8, z: 0 }), 'causeway deck should not be water');
    assert(!world.isInWater({ x: -20, y: 0, z: 0 }), 'downtown should be land');
    assert(world.isInWater({ x: -930, y: 0, z: -750 }), 'swamp pool should be water');
  });

  await suite.check('economy-wallet', () => {
    const economy = game.economy;
    assert(economy, 'economy system missing');
    const start = economy.cash;
    let events = 0;
    const off = game.events.on('economy:cashChanged', () => events++);
    economy.add(100, 'test');
    assert(economy.cash === start + 100, 'add failed');
    assert(!economy.trySpend(economy.cash + 1, 'test'), 'overspend allowed');
    assert(economy.trySpend(100, 'test'), 'spend refused');
    assert(economy.cash === start, 'balance not restored');
    assert(game.state.cash === start, 'state.cash not synced');
    off();
    assert(events === 2, `expected 2 cash events, got ${events}`);
  });

  await suite.check('damage-armor-death-respawn', async () => {
    player.teleport({ x: 918, y: 0, z: 100 }, Math.PI);
    const economy = game.economy;
    if (economy) economy.add(1000 - economy.cash, 'test');
    player.entity.health = 100;
    player.entity.armor = 100;
    player.applyDamage({ amount: 40, type: 'bullet' });
    assert(Math.abs(player.health - 80) < 1e-6 && Math.abs(player.armor - 80) < 1e-6, `armor absorb wrong: hp ${player.health} armor ${player.armor}`);
    let died = 0;
    let respawned = 0;
    const offA = game.events.on('player:died', () => died++);
    const offB = game.events.on('player:respawned', () => respawned++);
    player.applyDamage({ amount: 500, type: 'explosion' });
    assert(!player.isAlive && died === 1, 'player did not die');
    await waitFor(() => player.isAlive, 6000, 'respawn');
    offA();
    offB();
    assert(respawned === 1, 'player:respawned not emitted');
    const hospital = world.landmarks.nearestOfKind(player.position, 'hospital');
    assert(hospital && horizontalDistance(player.position, hospital.door) < 3, 'did not respawn at the hospital door');
    if (economy) assert(economy.cash === 900, `expected 10% cash loss, cash is ${economy.cash}`);
    assert(player.health === 100, 'health not restored');
  });

  await suite.check('interiors-enter-exit', async () => {
    const interiors = game.interiors;
    assert(interiors, 'interiors system missing');
    const entered: string[] = [];
    const exited: string[] = [];
    game.events.on('interior:entered', (e) => entered.push(e.id));
    game.events.on('interior:exited', (e) => exited.push(e.id));
    interiors.register({
      id: 'store_downtown',
      spawnPoint: { x: 0, y: 0.05, z: 3 },
      spawnHeading: Math.PI,
      build: (ctx) => {
        const floor = new Mesh(new BoxGeometry(12, 0.2, 12), new MeshStandardMaterial({ color: 0x777777 }));
        floor.position.y = -0.1;
        ctx.root.add(floor);
        ctx.physics.addStaticCuboid(ctx.toWorld({ x: 0, y: -0.1, z: 0 }), { x: 6, y: 0.1, z: 6 }, { groups: Groups.STATIC });
        return { root: floor, dispose: () => floor.geometry.dispose() };
      },
    });
    assert(await interiors.enter('store_downtown'), 'enter refused');
    assert(game.state.interior === 'store_downtown' && entered[0] === 'store_downtown', 'interior state/event wrong');
    assert(player.position.y > 500, 'player not moved to the interior');
    await waitFrames(game, 5);
    assert(await interiors.exit(), 'exit refused');
    assert(game.state.interior === null && exited[0] === 'store_downtown', 'exit state/event wrong');
    const door = world.landmarks.byId('store_downtown');
    assert(door && horizontalDistance(player.position, door.door) < 3, 'player not back at the store door');
  });

  await suite.check('camera-modes-and-cinematic', async () => {
    const cam = game.camera;
    cam.setMode('aim');
    await waitFrames(game, 3);
    assert(cam.mode === 'aim' || cam.mode === 'thirdPerson', 'aim mode rejected');
    cam.setMode('thirdPerson');
    const p = player.position;
    await cam.cinematic.moveTo({ x: p.x + 10, y: 20, z: p.z + 10 }, p, 0.3);
    const modeDuring: string = cam.mode;
    assert(modeDuring === 'cinematic', 'cinematic mode not active');
    assert(cam.three.position.distanceTo(cam.position) < 1, 'view camera not following the smoothed state');
    cam.cinematic.stop();
    assert(cam.mode === 'thirdPerson', `mode after cinematic stop: ${cam.mode}`);
    cam.shake(0.3, 0.2);
    cam.fovKick(5);
  });

  await suite.check('save-load-roundtrip', () => {
    const before = game.save.snapshot();
    assert('player' in before && 'economy' in before, 'player/economy not registered with save');
    assert(game.save.save('autotest'), 'save failed');
    assert(game.save.hasSave('autotest'), 'hasSave false after save');
    assert(game.save.load('autotest'), 'load failed');
    game.save.clear('autotest');
  });

  await suite.check('no-errors-after-run', () => {
    assert(game.debug.errorCount === 0, `${game.debug.errorCount} console errors: ${game.debug.errors.join(' | ')}`);
  });

  // Leave the camera on the Vice Beach sunset for the screenshot.
  game.time.setHour(18.5);
  player.teleport({ x: 575, y: 0, z: 40 }, -Math.PI / 2);
  game.camera.setOrbit(-Math.PI / 2 - 0.35, 0.12);
  await waitFrames(game, 3);
  suite.finish();
}
