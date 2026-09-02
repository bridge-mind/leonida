import { describe, expect, it } from 'vitest';
import { LuckyWheel, WHEEL_SEGMENTS, WHEEL_SEGMENT_COUNT, rollMystery, segmentAngle } from '../../src/casino/LuckyWheel';
import { Random } from '../../src/core/Random';

describe('LuckyWheel', () => {
  it('has 20 segments covering every prize kind with one vehicle', () => {
    expect(WHEEL_SEGMENT_COUNT).toBe(20);
    const kinds = WHEEL_SEGMENTS.map((s) => s.prize.kind);
    expect(kinds.filter((k) => k === 'vehicle')).toHaveLength(1);
    expect(kinds).toContain('mystery');
    expect(kinds).toContain('chips');
    const cash = WHEEL_SEGMENTS.filter((s) => s.prize.kind === 'cash').map((s) => (s.prize.kind === 'cash' ? s.prize.amount : 0));
    expect(Math.min(...cash)).toBe(500);
    expect(Math.max(...cash)).toBe(10000);
    const chips = WHEEL_SEGMENTS.filter((s) => s.prize.kind === 'chips').map((s) => (s.prize.kind === 'chips' ? s.prize.amount : 0));
    expect(new Set(chips)).toEqual(new Set([1000, 2500]));
    for (const s of WHEEL_SEGMENTS) expect(s.weight).toBeGreaterThan(0);
    expect(segmentAngle(0)).toBeCloseTo(Math.PI / 20);
  });

  it('allows one spin per game day', () => {
    const wheel = new LuckyWheel(() => 0.5);
    expect(wheel.canSpin(1)).toBe(true);
    const first = wheel.spin(1);
    expect(first).not.toBeNull();
    expect(wheel.canSpin(1)).toBe(false);
    expect(wheel.spin(1)).toBeNull();
    expect(wheel.canSpin(2)).toBe(true);
    expect(wheel.spin(2)).not.toBeNull();
    expect(wheel.lastSpinDay).toBe(2);
  });

  it('honours a forced segment index and ignores invalid ones', () => {
    const wheel = new LuckyWheel(() => 0.5);
    const spin = wheel.spin(3, 10);
    expect(spin?.index).toBe(10);
    expect(spin?.segment.prize.kind).toBe('vehicle');
    const bad = wheel.spin(4, 99);
    expect(bad).not.toBeNull();
    expect(bad && bad.index >= 0 && bad.index < 20).toBe(true);
  });

  it('weighted picks favour cheap prizes and rarely hit the car', () => {
    const rng = new Random(9);
    const wheel = new LuckyWheel(() => rng.next());
    const counts = new Array<number>(WHEEL_SEGMENT_COUNT).fill(0);
    const n = 20000;
    for (let i = 0; i < n; i++) counts[wheel.pickIndex()]++;
    const totalWeight = WHEEL_SEGMENTS.reduce((s, seg) => s + seg.weight, 0);
    for (let i = 0; i < WHEEL_SEGMENT_COUNT; i++) {
      const expected = (WHEEL_SEGMENTS[i].weight / totalWeight) * n;
      expect(Math.abs(counts[i] - expected)).toBeLessThan(expected * 0.3 + 40);
    }
    const vehicleIndex = WHEEL_SEGMENTS.findIndex((s) => s.prize.kind === 'vehicle');
    expect(counts[vehicleIndex] / n).toBeLessThan(0.02);
    expect(rollMystery(() => 0.1)).toEqual({ kind: 'weapon', weaponId: 'smg', ammo: 60 });
    expect(rollMystery(() => 0.5).kind).toBe('ammo');
    expect(rollMystery(() => 0.9)).toEqual({ kind: 'armor', amount: 100 });
  });
});
