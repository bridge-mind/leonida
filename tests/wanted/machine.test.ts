import { describe, expect, it } from 'vitest';
import type { WantedState } from '../../src/core/GameState';
import { CRIME_TABLE, dispatchTier, MAX_WANTED_LEVEL } from '../../src/data/police';
import { WantedConfig } from '../../src/systems/wanted/config';
import { WantedMachine, type WantedHooks } from '../../src/systems/wanted/machine';

const DT = 1 / 60;
const ORIGIN = { x: 0, y: 0, z: 0 };
const NONE = { copSaw: false, civilianSaw: false };
const CIVILIAN = { copSaw: false, civilianSaw: true };
const COP = { copSaw: true, civilianSaw: false };

interface Log {
  changed: { level: number; state: WantedState; previousLevel: number }[];
  cleared: string[];
  stars: number[];
  searching: number;
  reraised: number;
}

function make(): { m: WantedMachine; log: Log } {
  const log: Log = { changed: [], cleared: [], stars: [], searching: 0, reraised: 0 };
  const hooks: WantedHooks = {
    changed: (level, state, previousLevel) => log.changed.push({ level, state, previousLevel }),
    cleared: (reason) => log.cleared.push(reason),
    starsGained: (n) => log.stars.push(n),
    searching: () => log.searching++,
    hotSceneReraised: () => log.reraised++,
  };
  return { m: new WantedMachine(hooks), log };
}

function run(m: WantedMachine, seconds: number, pos = ORIGIN, seen = false): void {
  const steps = Math.round(seconds / DT);
  for (let i = 0; i < steps; i++) m.tick(DT, pos, seen);
}

describe('crime table', () => {
  it('ignores an unwitnessed quiet crime at level 0', () => {
    const { m, log } = make();
    m.tick(DT, ORIGIN, false);
    const r = m.reportCrime('shot_fired', ORIGIN, NONE);
    expect(r.counted).toBe(false);
    expect(m.level).toBe(0);
    expect(log.changed).toHaveLength(0);
  });

  it('civilian witness raises to 1 responding, cop witness to active', () => {
    const a = make();
    a.m.tick(DT, ORIGIN, false);
    a.m.reportCrime('shot_fired', ORIGIN, CIVILIAN);
    expect(a.m.level).toBe(1);
    expect(a.m.state).toBe('responding');
    expect(a.m.hasSearchCenter).toBe(true);
    expect(a.log.changed[0]).toEqual({ level: 1, state: 'responding', previousLevel: 0 });

    const b = make();
    b.m.tick(DT, ORIGIN, false);
    b.m.reportCrime('carjack', ORIGIN, COP);
    expect(b.m.state).toBe('active');
  });

  it('applies the table values and caps at 6', () => {
    const { m } = make();
    m.reportCrime('rob_store', ORIGIN, NONE);
    expect(m.level).toBe(2);
    m.reportCrime('kill_cop', ORIGIN, NONE);
    expect(m.level).toBe(4);
    m.reportCrime('explosion', ORIGIN, NONE);
    expect(m.level).toBe(6);
    m.reportCrime('kill_cop', ORIGIN, NONE);
    expect(m.level).toBe(MAX_WANTED_LEVEL);
  });

  it('shot_fired while wanted resets the timer but adds no star; kill_ped adds 2', () => {
    const { m } = make();
    m.reportCrime('shot_fired', ORIGIN, CIVILIAN);
    run(m, 5);
    expect(m.evasion).toBeGreaterThan(0);
    m.reportCrime('shot_fired', ORIGIN, NONE);
    expect(m.level).toBe(1);
    expect(m.evasion).toBe(0);
    m.reportCrime('kill_ped', ORIGIN, NONE);
    expect(m.level).toBe(3);
  });

  it('honours per-crime cooldowns (ramming a cruiser)', () => {
    const { m } = make();
    expect(m.reportCrime('hit_cop_vehicle', ORIGIN, NONE).counted).toBe(true);
    expect(m.reportCrime('hit_cop_vehicle', ORIGIN, NONE).counted).toBe(false);
    run(m, CRIME_TABLE.hit_cop_vehicle.cooldown + 0.1);
    expect(m.reportCrime('hit_cop_vehicle', ORIGIN, NONE).counted).toBe(true);
    expect(m.level).toBe(1);
  });

  it('adds 5 heat per star and decays 1 per minute', () => {
    const { m } = make();
    m.reportCrime('rob_store', ORIGIN, NONE);
    expect(m.heat).toBe(10);
    m.clear('debug');
    run(m, 60);
    expect(m.heat).toBeCloseTo(9, 1);
  });
});

describe('state machine', () => {
  it('responding → active when seen, → searching after losing sight, timer resets on re-acquire', () => {
    const { m, log } = make();
    m.reportCrime('shot_fired', ORIGIN, CIVILIAN);
    run(m, 1, ORIGIN, true);
    expect(m.state).toBe('active');
    run(m, WantedConfig.vision.loseSightAfter + 0.2);
    expect(m.state).toBe('searching');
    expect(log.searching).toBe(1);
    run(m, 5);
    expect(m.evasion).toBeGreaterThan(0);
    run(m, 0.5, ORIGIN, true);
    expect(m.state).toBe('active');
    expect(m.evasion).toBe(0);
  });

  it('search radius grows 8 m/s from 40 m and caps at 220 + 30·level', () => {
    const { m } = make();
    m.setLevel(2, 'searching');
    expect(m.searchRadius).toBe(WantedConfig.search.initialRadius);
    const far = { x: 1000, y: 0, z: 0 };
    run(m, 10, far);
    expect(m.searchRadius).toBeCloseTo(40 + 80, 0);
    m.evasion = 0;
    run(m, 30, far);
    expect(m.searchRadius).toBe(220 + 60);
  });

  it('clears after level×20+20 s unseen outside the circle, 1.5× slower inside', () => {
    const outside = make();
    outside.m.setLevel(1, 'searching');
    run(outside.m, 39.9, { x: 500, y: 0, z: 0 });
    expect(outside.m.level).toBe(1);
    run(outside.m, 0.3, { x: 500, y: 0, z: 0 });
    expect(outside.m.level).toBe(0);
    expect(outside.m.state).toBe('none');
    expect(outside.log.cleared).toEqual(['evaded']);
    expect(outside.log.changed.at(-1)).toEqual({ level: 0, state: 'none', previousLevel: 1 });

    const inside = make();
    inside.m.setLevel(1, 'searching');
    run(inside.m, 45, ORIGIN);
    expect(inside.m.level).toBe(1);
    run(inside.m, 16, ORIGIN);
    expect(inside.m.level).toBe(0);
  });

  it('never clears while seen', () => {
    const { m } = make();
    m.setLevel(1);
    run(m, 120, ORIGIN, true);
    expect(m.level).toBe(1);
    expect(m.state).toBe('active');
  });

  it('phone report raises to at least 1 without lowering a higher level', () => {
    const { m } = make();
    m.reportedByPhone({ x: 10, y: 0, z: 0 });
    expect(m.level).toBe(1);
    expect(m.state).toBe('responding');
    expect(m.searchCenter.x).toBe(10);
    m.setLevel(3);
    m.reportedByPhone(ORIGIN);
    expect(m.level).toBe(3);
    expect(m.state).toBe('active');
  });

  it('clear emits cleared then changed and resets the circle', () => {
    const { m, log } = make();
    m.setLevel(3);
    m.clear('busted');
    expect(log.cleared).toEqual(['busted']);
    expect(m.hasSearchCenter).toBe(false);
    expect(m.searchRadius).toBe(0);
    expect(log.changed.at(-1)).toEqual({ level: 0, state: 'none', previousLevel: 3 });
    m.clear('again');
    expect(log.changed).toHaveLength(2);
  });
});

describe('hot scenes', () => {
  it('re-raises only on an outside→inside transition and expires after 120 s', () => {
    const { m, log } = make();
    m.tick(DT, ORIGIN, false);
    m.reportCrime('shot_fired', ORIGIN, CIVILIAN);
    expect(m.hotScenes).toHaveLength(1);
    m.clear('debug');
    run(m, 2, ORIGIN);
    expect(m.level).toBe(0);
    run(m, 1, { x: 200, y: 0, z: 0 });
    run(m, 1, ORIGIN);
    expect(m.level).toBe(1);
    expect(m.state).toBe('searching');
    expect(log.reraised).toBe(1);
    m.clear('debug');
    run(m, WantedConfig.hotScene.expirySeconds, { x: 200, y: 0, z: 0 });
    expect(m.hotScenes).toHaveLength(0);
    run(m, 1, ORIGIN);
    expect(m.level).toBe(0);
  });

  it('merges nearby crimes into one scene and caps the list', () => {
    const { m } = make();
    m.reportCrime('shot_fired', ORIGIN, CIVILIAN);
    m.reportCrime('shot_fired', { x: 5, y: 0, z: 5 }, NONE);
    expect(m.hotScenes).toHaveLength(1);
    for (let i = 0; i < 20; i++) m.reportCrime('shot_fired', { x: i * 100, y: 0, z: 0 }, NONE);
    expect(m.hotScenes.length).toBeLessThanOrEqual(WantedConfig.hotScene.max);
  });
});

describe('dispatch table', () => {
  it('escalates per level and clamps', () => {
    expect(dispatchTier(0).cruisers).toBe(0);
    expect(dispatchTier(1)).toMatchObject({ cruisers: 1, lethal: false });
    expect(dispatchTier(3).roadblock).toBe(true);
    expect(dispatchTier(4)).toMatchObject({ swatVans: 1, helicopters: 1 });
    expect(dispatchTier(5)).toMatchObject({ spikes: true, helicopters: 2 });
    expect(dispatchTier(9)).toEqual(dispatchTier(6));
  });
});
