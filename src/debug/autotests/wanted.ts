import type { Game } from '../../core/Game';
import type { WantedState } from '../../core/GameState';
import type { PedSystem } from '../../systems/peds';
import { assert, Suite, waitFor, waitFrames } from './util';

interface WantedDebugState {
  level: number;
  state: WantedState;
  heat: number;
  seen: boolean;
  evasion: number;
  evasionSeconds: number;
  searchRadius: number;
  hotScenes: number;
}

/**
 * Wanted-level acceptance (TEAM-PLAN §5 Agent 8).
 * Run: npm run smoke -- --systems=world,player,camera,vehicles,peds,weapons,damage,wanted,police,ui --name=wanted --autotest=wanted
 */
export default async function wanted(game: Game): Promise<void> {
  const suite = new Suite('wanted', game);
  const w = game.wanted;
  assert(w, 'wanted system missing');
  const peds = game.systems.get<PedSystem>('peds');
  const player = game.player;
  const run = (path: string, ...args: unknown[]): unknown => game.debug.run(path, ...args);
  const status = (): WantedDebugState => run('wanted.state') as WantedDebugState;
  const hasPolice = game.systems.has('police');
  const clearAll = (): void => {
    run('wanted.clear');
    if (hasPolice) run('police.clear');
  };

  game.time.setHour(12);
  player.teleport({ x: -20, y: 0, z: 120 }, Math.PI);
  game.camera.setOrbit(player.heading + Math.PI, 0.25);
  // Unarmed: level-1 cops try to arrest instead of shooting the test subject.
  game.weapons?.equip('fists');
  clearAll();
  await waitFrames(game, 5);

  await suite.check('witnessed-shot-raises-level-1-responding', async () => {
    assert(peds, 'peds system missing');
    run('peds.spawnHere', 'business');
    await waitFrames(game, 2);
    let changed = 0;
    const off = game.events.on('wanted:changed', () => changed++);
    const p = player.position;
    game.events.emit('crime:committed', { type: 'shot_fired', position: { x: p.x, y: p.y, z: p.z }, actor: 'player', loud: true });
    await waitFor(() => w.level >= 1, 1000, `level 1 (level ${w.level})`);
    off();
    assert(w.state === 'responding', `expected state responding, got ${w.state}`);
    assert(changed >= 1, 'wanted:changed not emitted');
    assert(game.state.wanted.level === 1 && game.state.wanted.state === 'responding', 'game.state.wanted not published');
    assert(status().hotScenes >= 1, 'crime did not create a hot scene');
    assert(w.heat > 0, 'heat did not rise with the star');
  });

  await suite.check('cop-line-of-sight-makes-active', async () => {
    assert(hasPolice, 'police system missing');
    // 20 m out: far enough that the arrest walk-up cannot finish before the next check.
    const p = player.position;
    const h = player.heading;
    const id = run('police.spawnCop', p.x + Math.sin(h) * 20, p.z + Math.cos(h) * 20) as number | null;
    assert(id !== null, 'police.spawnCop returned null');
    await waitFor(() => w.state === 'active', 2000, `state active (state ${w.state})`);
    assert(w.playerSeen, 'playerSeen false while a cop has line of sight');
  });

  await suite.check('losing-sight-enters-searching-then-evades', async () => {
    run('wanted.set', 2);
    if (hasPolice) run('police.clear');
    let cleared: string | null = null;
    const off = game.events.on('wanted:cleared', (e) => (cleared = e.reason));
    // No cops left: sight decays to `searching` after the lose-sight window.
    await waitFor(() => w.state === 'searching', 6000, `state searching (state ${w.state})`);
    const s0 = status();
    assert(game.state.wanted.searchCenter !== null, 'search circle not published while searching');
    run('wanted.fastForward', 20);
    const s1 = status();
    assert(s1.searchRadius > s0.searchRadius, `search radius did not grow (${s0.searchRadius} → ${s1.searchRadius})`);
    assert(s1.evasion >= 20, `evasion timer did not advance (${s1.evasion})`);
    run('wanted.fastForward', s1.evasionSeconds);
    off();
    assert(w.level === 0 && w.state === 'none', `expected cleared, got level ${w.level} state ${w.state}`);
    assert(cleared === 'evaded', `wanted:cleared reason ${cleared}, expected evaded`);
    assert(game.state.wanted.level === 0 && game.state.wanted.searchCenter === null, 'state.wanted not reset on clear');
  });

  await suite.check('hot-scene-reentry-reraises', async () => {
    assert(status().hotScenes >= 1, 'no hot scene left from the earlier crime');
    const scene = { x: player.position.x, y: player.position.y, z: player.position.z };
    // Leave the 60 m scene, then walk back in: outside→inside re-raises to 1 star, searching.
    player.teleport({ x: scene.x, y: 0, z: scene.z - 140 }, Math.PI);
    await waitFrames(game, 4);
    assert(w.level === 0, `level rose while outside the scene (${w.level})`);
    player.teleport(scene, Math.PI);
    await waitFor(() => w.level === 1, 1000, `hot scene re-raise (level ${w.level})`);
    assert(w.state === 'searching', `expected searching after re-entry, got ${w.state}`);
    clearAll();
    await waitFrames(game, 2);
    // Standing inside a scene after a clear must not re-trigger until the player leaves.
    assert(w.level === 0, 'hot scene re-raised without leaving first');
  });

  await suite.check('phone-report-raises-responding', async () => {
    assert(peds, 'peds system missing');
    const id = run('peds.spawnHere', 'tourist') as number;
    const ped = peds.byId(id);
    assert(ped, 'spawned ped not found');
    const p = ped.position;
    game.events.emit('ped:reportedCrime', { ped, position: { x: p.x, y: p.y, z: p.z } });
    await waitFor(() => w.level >= 1, 500, 'phone report to raise the level');
    assert(w.state === 'responding', `expected responding, got ${w.state}`);
    const c = game.state.wanted.searchCenter;
    assert(c && Math.abs(c.x - p.x) < 1 && Math.abs(c.z - p.z) < 1, 'search center not at the reported crime');
    clearAll();
  });

  await suite.check('crime-while-wanted-adds-stars-and-caps', async () => {
    const level = (): number => status().level;
    run('wanted.set', 5, 'active');
    w.reportCrime('kill_cop', player.position, 'player');
    assert(level() === 6, `kill_cop at 5 should cap at 6, got ${level()}`);
    w.reportCrime('explosion', player.position, 'player');
    assert(level() === 6, `level exceeded the cap (${level()})`);
    run('wanted.set', 1, 'active');
    w.reportCrime('shot_fired', player.position, 'player');
    assert(level() === 1, `shot_fired while wanted should not add stars (level ${level()})`);
    w.reportCrime('kill_ped', player.position, 'player');
    assert(level() === 3, `kill_ped while wanted should add 2 (level ${level()})`);
    clearAll();
  });

  await suite.check('respawn-clears-level', async () => {
    run('wanted.set', 2, 'active');
    game.events.emit('player:respawned', { position: { x: 0, y: 0, z: 0 }, reason: 'wasted' });
    await waitFrames(game, 1);
    assert(w.level === 0, `level ${w.level} after respawn`);
    assert(game.debug.errorCount === 0, `${game.debug.errorCount} console errors: ${game.debug.errors.join(' | ')}`);
  });

  clearAll();
  suite.finish();
}
