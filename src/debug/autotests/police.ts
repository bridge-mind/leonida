import type { Game } from '../../core/Game';
import type { PoliceSystem } from '../../systems/police';
import type { UnitsReport } from '../../systems/police/debug';
import { assert, Suite, waitFor, waitFrames } from './util';

/**
 * Police response acceptance (TEAM-PLAN §5 Agent 8).
 * Run: npm run smoke -- --systems=world,player,camera,vehicles,traffic,peds,weapons,damage,particles,wanted,police,ui --name=police --autotest=police
 */
export default async function police(game: Game): Promise<void> {
  const suite = new Suite('police', game);
  const sys = game.systems.get<PoliceSystem>('police');
  assert(sys, 'police system missing');
  const w = game.wanted;
  assert(w, 'wanted system missing');
  const vehicles = game.vehicles;
  assert(vehicles, 'vehicles system missing');
  const player = game.player;
  const run = (path: string, ...args: unknown[]): unknown => game.debug.run(path, ...args);
  const units = (): UnitsReport => run('police.units') as UnitsReport;
  const reset = (): void => {
    run('wanted.clear');
    run('police.clear');
  };

  game.time.setHour(12);
  player.teleport({ x: -20, y: 0, z: 120 }, Math.PI);
  game.camera.setOrbit(player.heading + Math.PI, 0.25);
  game.weapons?.equip('fists');
  reset();
  await waitFrames(game, 5);

  await suite.check('level-4-dispatches-swat-and-helicopter', async () => {
    run('wanted.set', 4);
    await waitFor(() => units().helicopters >= 1, 12000, `a helicopter (units ${JSON.stringify(units())})`);
    await waitFor(() => vehicles.all.some((v) => v.archetypeId === 'swat'), 12000, 'a swat van in vehicles.all');
    await waitFor(() => units().cruisers >= 1, 6000, 'a cruiser');
    const u = units();
    assert(u.cops >= 2, `only ${u.cops} cops for level 4`);
    assert(sys.activeUnits >= 2, `activeUnits ${sys.activeUnits}`);
    const cruiser = vehicles.all.find((v) => v.archetypeId === 'police' && v.driver !== null);
    assert(cruiser, 'no cruiser with a PoliceDriver');
    assert(cruiser.state.sirenOn, 'cruiser siren off');
    const scratch = sys.copsNear(player.position, 400, []);
    assert(scratch.length >= 2, `copsNear found ${scratch.length}`);
    assert(scratch.every((c) => c.isCop), 'copsNear returned a non-cop');
  });

  await suite.check('clear-stands-units-down', async () => {
    run('wanted.clear');
    await waitFrames(game, 2);
    const cruiser = vehicles.all.find((v) => v.archetypeId === 'police' && v.driver !== null);
    assert(!cruiser || !cruiser.state.sirenOn, 'siren still on after the level cleared');
    assert(sys.helicopters.every((h) => h.leaving), 'helicopter not leaving after clear');
    await waitFor(() => units().helicopters === 0, 25000, 'helicopters to leave');
    run('police.clear');
    assert(units().cops === 0 && units().cruisers === 0, `units remain after police.clear: ${JSON.stringify(units())}`);
  });

  await suite.check('helicopter-takes-damage-and-crashes', async () => {
    run('wanted.set', 4);
    const id = run('police.heli') as number | null;
    assert(id !== null, 'police.heli returned null');
    const heli = sys.helicopters.find((h) => h.id === id);
    assert(heli, 'spawned helicopter not tracked');
    const before = w.level;
    heli.applyDamage({ amount: 50, type: 'bullet', source: 'player' });
    assert(w.level >= before, 'attack_helicopter lowered the level');
    heli.applyDamage({ amount: 1000, type: 'bullet', source: 'player' });
    assert(!heli.isAlive, 'helicopter alive after 1050 damage (600 HP)');
    await waitFor(() => !sys.helicopters.includes(heli), 8000, 'wreck to explode and be removed');
    reset();
  });

  await suite.check('roadblock-spawns-across-the-road-ahead', async () => {
    const graph = game.world.roadGraph;
    const lp = graph.nearestLanePoint(player.position);
    assert(lp, 'no lane near the player');
    // Stand on the lane facing along it so a road exists 120–200 m ahead.
    player.teleport({ x: lp.point.x, y: lp.point.y, z: lp.point.z }, Math.atan2(lp.direction.x, lp.direction.z));
    await waitFrames(game, 2);
    run('wanted.set', 3);
    const ok = run('police.roadblock', true) as boolean;
    assert(ok, 'police.roadblock could not place a block');
    const u = units();
    assert(u.roadblocks >= 1, 'roadblock not counted');
    assert(u.footCops >= 2, `roadblock spawned ${u.footCops} foot cops`);
    const parked = vehicles.all.filter((v) => v.archetypeId === 'police' && v.driver === null && v.state.sirenOn);
    assert(parked.length >= 2, `expected 2 parked cruisers, found ${parked.length}`);
    reset();
    await waitFrames(game, 2);
    assert(units().roadblocks === 0, 'roadblock survived clear');
  });

  await suite.check('level-1-cop-adjacent-busts-standing-player', async () => {
    player.teleport({ x: -20, y: 0, z: 120 }, Math.PI);
    await waitFrames(game, 2);
    let busted = false;
    let respawned: 'wasted' | 'busted' | null = null;
    const offs = [
      game.events.on('player:busted', () => (busted = true)),
      game.events.on('player:respawned', (e) => (respawned = e.reason)),
    ];
    run('wanted.set', 1, 'active');
    const p = player.position;
    const h = player.heading;
    const id = run('police.spawnCop', p.x + Math.sin(h) * 2, p.z + Math.cos(h) * 2, 'pistol') as number | null;
    assert(id !== null, 'police.spawnCop returned null');
    try {
      await waitFor(() => busted, 6000, `player:busted (arrest timer ${units().arrestTimer.toFixed(2)}, level ${w.level})`);
      await waitFor(() => respawned !== null, 8000, 'player:respawned after the bust');
    } finally {
      offs.forEach((off) => off());
    }
    assert(respawned === 'busted', `respawn reason ${respawned}`);
    assert(w.level === 0, `level ${w.level} after the bust`);
    assert(player.isAlive, 'player not alive after respawn');
  });

  await suite.check('no-console-errors', () => {
    assert(game.debug.errorCount === 0, `${game.debug.errorCount} console errors: ${game.debug.errors.join(' | ')}`);
  });

  reset();
  suite.finish();
}
