import type { Game } from '../../core/Game';
import { assert, Suite, wait, waitFor, waitFrames } from './util';

interface MissionListEntry {
  id: string;
  name: string;
  state: 'done' | 'available' | 'locked' | 'active';
}

/**
 * Missions acceptance (docs/TEAM-PLAN.md Agent 10).
 * Run: npm run smoke -- --systems=world,player,camera,ui,vehicles,peds,weapons,damage,pickups,navigation,economy,wanted,police,missions,cutscene,companion,characters --name=missions --autotest=missions
 * Everything is driven through debug actions so it also runs headless with only core systems.
 */
export default async function missions(game: Game): Promise<void> {
  const suite = new Suite('missions', game);
  assert(game.systems.has('missions'), 'missions system missing');
  const peer = game.missions;
  assert(peer, 'game.missions peer missing');
  const errorsAtStart = game.debug.errors.length;
  const list = (): MissionListEntry[] => game.debug.run('missions.list') as MissionListEntry[];
  const stateOf = (id: string): string | undefined => list().find((m) => m.id === id)?.state;
  const cash = (): number => game.economy?.cash ?? 0;

  game.debug.run('missions.reset');
  await waitFrames(game, 2);

  await suite.check('list-and-unlocks', () => {
    const all = list();
    assert(all.length >= 10, `expected 8 story missions + 2 jobs, got ${all.length}`);
    assert(stateOf('m1') === 'available', `m1 should be available, got ${stateOf('m1')}`);
    assert(stateOf('m2') === 'locked', `m2 should be locked, got ${stateOf('m2')}`);
    assert(peer.available().some((m) => m.id === 'm1'), 'available() misses m1');
  });

  await suite.check('m1-fastforward-to-passed', async () => {
    const got: { passed: { id: string; reward: number } | null; started: boolean } = { passed: null, started: false };
    const offs = [
      game.events.on('mission:started', (e) => (got.started ||= e.id === 'm1')),
      game.events.on('mission:passed', (e) => (got.passed = e)),
    ];
    const before = cash();
    try {
      assert(peer.start('m1'), 'start(m1) returned false');
      await waitFrames(game, 2);
      assert(got.started, 'mission:started not emitted');
      assert(peer.active === 'm1', `active is ${peer.active}`);
      assert(game.state.objective, 'objective text not published');
      for (let i = 0; i < 40 && peer.active; i++) {
        game.debug.run('missions.fastForward');
        await waitFrames(game, 2);
      }
      await waitFor(() => got.passed !== null, 15000, 'mission:passed m1');
      const passed = got.passed;
      assert(passed && passed.id === 'm1', 'passed event is not m1');
      assert(peer.completed.includes('m1'), 'completed() misses m1');
      if (game.economy) assert(cash() === before + passed.reward, `cash ${cash()}, expected ${before + passed.reward}`);
      assert(stateOf('m2') === 'available', 'm2 did not unlock after m1');
    } finally {
      for (const off of offs) off();
    }
  });

  await suite.check('m2-death-fails-and-retry-prompt', async () => {
    const got: { failed: { id: string; reason: string } | null } = { failed: null };
    const off = game.events.on('mission:failed', (e) => (got.failed = e));
    try {
      assert(peer.start('m2'), 'start(m2) returned false');
      // Skip the opening cutscene so the death lands on a real objective.
      await waitFrames(game, 2);
      game.debug.run('missions.fastForward');
      await waitFrames(game, 2);
      game.debug.run('player.kill');
      await waitFor(() => got.failed !== null, 5000, 'mission:failed');
      assert(got.failed?.reason === 'wasted', `fail reason ${got.failed?.reason}`);
      await waitFor(() => game.debug.run('missions.retryVisible') === true, 12000, 'retry prompt');
      game.debug.run('missions.retry', 'quit');
      await waitFrames(game, 2);
      assert(game.debug.run('missions.retryVisible') === false, 'retry prompt still visible');
      assert(!game.input.isBlocked, 'input blocker left behind by the retry prompt');
      assert(peer.active === null, 'mission still active after quit');
    } finally {
      off();
    }
  });

  for (const id of ['m2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'job_chopshop', 'job_robbery']) {
    await suite.check(`complete-${id}`, async () => {
      const errorsBefore = game.debug.errors.length;
      const status = (await game.debug.run('missions.complete', id)) as string;
      assert(status === 'passed', `${id} ended ${status}`);
      const newErrors = game.debug.errors.slice(errorsBefore);
      assert(newErrors.length === 0, `console errors during ${id}: ${newErrors.join(' | ')}`);
      assert(peer.active === null, 'run still active after complete');
    });
  }

  await suite.check('jobs-repeatable-after-finale', () => {
    assert(stateOf('job_chopshop') === 'available', `job_chopshop should be available, got ${stateOf('job_chopshop')}`);
    assert(stateOf('job_robbery') === 'available', `job_robbery should be available, got ${stateOf('job_robbery')}`);
    assert(stateOf('m8') === 'done', 'm8 should be done');
  });

  await suite.check('no-leaks', async () => {
    await wait(50);
    assert(!game.input.isBlocked, 'input still blocked');
    assert(game.time.scale === 1, `time scale ${game.time.scale}`);
    assert(!game.state.objective, `objective still shown: ${game.state.objective}`);
    const total = game.debug.errors.length - errorsAtStart;
    assert(total === 0, `${total} console errors during the suite`);
  });

  suite.finish();
}
