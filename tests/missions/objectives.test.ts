import { describe, expect, it } from 'vitest';
import type { Entity } from '../../src/core/entities';
import { action, custom, deliver, enterVehicle, goto, kill, survive, timed, waitFor } from '../../src/missions/objectives';
import { escapeWanted, reachWanted, robStore, winAtCasino } from '../../src/missions/objectivesWorld';
import type { MissionContext, MissionDef, Objective } from '../../src/missions/types';
import { MissionRun } from '../../src/systems/missions/Runner';
import { makeGame, makeVehicle } from './fakeGame';

const DT = 1 / 60;

function defOf(build: (ctx: MissionContext) => Objective[], reward = 100): MissionDef {
  return { id: 't', name: 'Test', contact: 'Nobody', position: { x: 0, y: 0, z: 0 }, reward, repeatable: false, build };
}

function run(game: ReturnType<typeof makeGame>['game'], objectives: (ctx: MissionContext) => Objective[], reward = 100): MissionRun {
  const r = new MissionRun(game, defOf(objectives, reward));
  r.start();
  return r;
}

function tick(r: MissionRun, n = 1): void {
  for (let i = 0; i < n; i++) r.update(DT);
}

describe('goto / deliver / enterVehicle', () => {
  it('completes when the player walks into the radius', () => {
    const { game, fake } = makeGame();
    const r = run(game, () => [goto({ x: 50, y: 0, z: 0 }, 3, 'Go')]);
    tick(r);
    expect(r.status).toBe('running');
    expect(r.text).toBe('Go');
    fake.player.position = { x: 48, y: 0, z: 1 };
    tick(r);
    expect(r.status).toBe('passed');
  });

  it('deliver fails when the vehicle is wrecked and needs the player inside it', () => {
    const { game, fake } = makeGame();
    const v = makeVehicle({ x: 0, y: 0, z: 0 });
    const r = run(game, () => [deliver(() => v, { x: 10, y: 0, z: 0 }, 3, 'Deliver')]);
    fake.player.position = { x: 10, y: 0, z: 0 };
    tick(r);
    expect(r.status).toBe('running');
    fake.player.mount(v);
    v.wrecked = true;
    tick(r);
    expect(r.status).toBe('failed');
    expect(r.failReason).toBe('vehicle destroyed');
  });

  it('enterVehicle completes on mount and fast-forward mounts', () => {
    const { game, fake } = makeGame();
    const v = makeVehicle({ x: 30, y: 0, z: 30 });
    const r = run(game, () => [enterVehicle(() => v, 'Get in')]);
    tick(r);
    expect(r.status).toBe('running');
    r.fastForward();
    expect(fake.player.mountedVehicle).toBe(v);
    expect(r.status).toBe('passed');
  });
});

describe('fail conditions', () => {
  it('player death fails the mission with "wasted" and runs cleanups', () => {
    const { game, fake } = makeGame();
    let cleaned = 0;
    const r = run(game, (ctx) => {
      ctx.onCleanup(() => cleaned++);
      return [survive(30, 'Survive')];
    });
    tick(r, 5);
    fake.events.emit('player:died', { position: { x: 0, y: 0, z: 0 } });
    expect(r.status).toBe('failed');
    expect(r.failReason).toBe('wasted');
    expect(cleaned).toBe(1);
  });

  it('busted fails with "busted"', () => {
    const { game, fake } = makeGame();
    const r = run(game, () => [survive(30, 'Survive')]);
    fake.events.emit('player:busted', { position: { x: 0, y: 0, z: 0 } });
    expect(r.status).toBe('failed');
    expect(r.failReason).toBe('busted');
  });

  it('timed wrapper fails with its reason when the clock runs out', () => {
    const { game } = makeGame();
    const r = run(game, () => [timed(goto({ x: 999, y: 0, z: 0 }, 1, 'Far'), 1, 'too slow')]);
    tick(r, 61);
    expect(r.status).toBe('failed');
    expect(r.failReason).toBe('too slow');
  });

  it('a throwing objective fails the run instead of crashing the loop', () => {
    const { game } = makeGame();
    const r = run(game, () => [
      custom({
        text: 'Boom',
        update: () => {
          throw new Error('bad script');
        },
        skip() {},
      }),
    ]);
    tick(r);
    expect(r.status).toBe('failed');
    expect(r.failReason).toBe('script error');
  });
});

describe('sequencing', () => {
  it('runs objectives in order, stops each once and passes at the end', () => {
    const { game, fake } = makeGame();
    const log: string[] = [];
    const step = (name: string, at: number): Objective =>
      custom({
        text: name,
        start: () => log.push(`start ${name}`),
        update: (ctx) => (ctx.game.player.position.x >= at ? 'done' : 'running'),
        skip: () => {},
        stop: () => log.push(`stop ${name}`),
      });
    const r = run(game, () => [
      step('a', 1),
      step('b', 2),
      action(() => {
        log.push('action');
      }),
    ]);
    tick(r);
    fake.player.position.x = 1;
    tick(r);
    expect(r.text).toBe('b');
    fake.player.position.x = 2;
    tick(r, 2);
    expect(r.status).toBe('passed');
    expect(log).toEqual(['start a', 'stop a', 'start b', 'stop b', 'action']);
  });

  it('fastForward walks a whole mission to passed', () => {
    const { game } = makeGame();
    const r = run(game, () => [goto({ x: 100, y: 0, z: 0 }, 2, 'A'), survive(60, 'B'), waitFor('casino:won', () => true, 'C')]);
    let guard = 0;
    while (r.status === 'running' && guard++ < 10) r.fastForward();
    expect(r.status).toBe('passed');
    expect(guard).toBe(3);
  });

  it('kill completes when every target is dead and survive ends early on orKill', () => {
    const { game } = makeGame();
    const targets = [{ isAlive: true }, { isAlive: true }] as unknown as (Entity & { isAlive: boolean })[];
    const r = run(game, () => [kill(() => targets, 'Kill'), survive(100, 'Hold', { orKill: () => targets })]);
    tick(r);
    expect(r.status).toBe('running');
    targets[0].isAlive = false;
    tick(r);
    expect(r.objectiveIndex).toBe(0);
    targets[1].isAlive = false;
    tick(r, 2);
    expect(r.status).toBe('passed');
  });

  it('abort stops the current objective and runs cleanups', () => {
    const { game } = makeGame();
    let stopped = 0;
    let cleaned = 0;
    const r = run(game, (ctx) => {
      ctx.onCleanup(() => cleaned++);
      return [custom({ text: 'x', update: () => 'running', skip() {}, stop: () => stopped++ })];
    });
    r.abort();
    expect(r.status).toBe('aborted');
    expect(stopped).toBe(1);
    expect(cleaned).toBe(1);
  });
});

describe('peer-backed objectives degrade gracefully', () => {
  it('escapeWanted and reachWanted complete at once without a wanted system', () => {
    const { game } = makeGame();
    const r = run(game, () => [reachWanted(4, 'Heat'), escapeWanted()]);
    tick(r, 2);
    expect(r.status).toBe('passed');
  });

  it('escapeWanted waits for level 0 when wanted exists', () => {
    const { game, fake } = makeGame({ wanted: true });
    fake.wanted!.level = 3;
    const r = run(game, () => [escapeWanted()]);
    tick(r);
    expect(r.status).toBe('running');
    fake.wanted!.level = 0;
    tick(r);
    expect(r.status).toBe('passed');
  });

  it('robStore adds the register cash to the reward', () => {
    const { game, fake } = makeGame();
    const r = run(game, () => [robStore('gasstation', null, 'Rob')], 300);
    fake.events.emit('robbery:completed', { storeId: 'other', amount: 999 });
    tick(r);
    expect(r.status).toBe('running');
    fake.events.emit('robbery:completed', { storeId: 'gasstation', amount: 450 });
    tick(r);
    expect(r.status).toBe('passed');
    expect(r.reward).toBe(750);
  });

  it('winAtCasino tracks net winnings and fails when broke', () => {
    const { game, fake } = makeGame({ casino: true });
    const r = run(game, () => [winAtCasino(2000, null, 'Win')]);
    fake.events.emit('casino:won', { game: 'blackjack', amount: 1500 });
    fake.events.emit('casino:lost', { game: 'blackjack', amount: 500 });
    tick(r);
    expect(r.status).toBe('running');
    expect(r.text).toContain('$1,000 / $2,000');
    fake.events.emit('casino:won', { game: 'blackjack', amount: 1000 });
    tick(r);
    expect(r.status).toBe('passed');

    const broke = run(game, () => [winAtCasino(2000, null, 'Win')]);
    fake.events.emit('casino:lost', { game: 'slots', amount: 100 });
    fake.casino!.chips = 0;
    tick(broke);
    expect(broke.status).toBe('failed');
    expect(broke.failReason).toBe('you went broke');
    expect(fake.ui!.notes.length).toBeGreaterThan(0);
  });
});
