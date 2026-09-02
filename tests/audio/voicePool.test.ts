import { describe, expect, it } from 'vitest';
import { VoicePool, type PoolVoice } from '../../src/systems/audio/voicePool';

class FakeVoice implements PoolVoice {
  killed = false;
  constructor(
    readonly loop: boolean,
    readonly startedAt: number,
    public volume: number,
  ) {}

  kill(): void {
    this.killed = true;
  }
}

describe('VoicePool', () => {
  it('never exceeds its caps when 200 voices are requested', () => {
    const pool = new VoicePool<FakeVoice>(40, 16);
    const all: FakeVoice[] = [];
    for (let i = 0; i < 200; i++) {
      const v = new FakeVoice(i % 5 === 0, i, Math.random());
      all.push(v);
      pool.add(v);
      expect(pool.oneShots.length).toBeLessThanOrEqual(40);
      expect(pool.loops.length).toBeLessThanOrEqual(16);
    }
    expect(pool.active).toBe(56);
    const alive = all.filter((v) => !v.killed);
    expect(alive).toHaveLength(56);
    expect(pool.stolen).toBe(200 - 56);
    for (const v of all) expect(v.killed || pool.oneShots.includes(v) || pool.loops.includes(v)).toBe(true);
  });

  it('steals the quietest voice first, then the oldest', () => {
    const pool = new VoicePool<FakeVoice>(3, 1);
    const loud = new FakeVoice(false, 0, 1);
    const quiet = new FakeVoice(false, 1, 0.1);
    const mid = new FakeVoice(false, 2, 0.5);
    pool.add(loud);
    pool.add(quiet);
    pool.add(mid);
    pool.add(new FakeVoice(false, 3, 0.9));
    expect(quiet.killed).toBe(true);
    expect(loud.killed).toBe(false);
    const older = new FakeVoice(true, 0, 0.5);
    const newer = new FakeVoice(true, 5, 0.5);
    pool.add(older);
    pool.add(newer);
    expect(older.killed).toBe(true);
    expect(pool.loops).toEqual([newer]);
  });

  it('prunes expired voices and removes idempotently', () => {
    const pool = new VoicePool<FakeVoice>(10, 10);
    const a = new FakeVoice(false, 0, 1);
    const b = new FakeVoice(true, 0, 1);
    pool.add(a);
    pool.add(b);
    pool.prune((v) => v === a);
    expect(a.killed).toBe(true);
    expect(pool.active).toBe(1);
    expect(pool.remove(b)).toBe(true);
    expect(pool.remove(b)).toBe(false);
    expect(pool.active).toBe(0);
    pool.add(b);
    pool.killAll();
    expect(b.killed).toBe(true);
    expect(pool.active).toBe(0);
  });
});
