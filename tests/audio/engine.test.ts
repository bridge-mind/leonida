import { describe, expect, it } from 'vitest';
import { AudioConfig } from '../../src/systems/audio/config';
import { EngineModel, engineHz, gearFor, gearFraction } from '../../src/systems/audio/engine';

describe('engine rpm mapping', () => {
  it('maps rpm01 to a monotonically increasing frequency', () => {
    let last = -Infinity;
    for (let i = 0; i <= 100; i++) {
      const hz = engineHz(55, i / 100);
      expect(hz).toBeGreaterThan(last);
      last = hz;
    }
  });

  it('spans idle to idle × (1 + rpmRange)', () => {
    expect(engineHz(55, 0)).toBeCloseTo(55);
    expect(engineHz(55, 1)).toBeCloseTo(55 * (1 + AudioConfig.engine.rpmRange));
    expect(engineHz(55, 1)).toBeCloseTo(55 * 3.5);
  });

  it('clamps rpm outside 0..1', () => {
    expect(engineHz(40, -1)).toBeCloseTo(40);
    expect(engineHz(40, 5)).toBeCloseTo(engineHz(40, 1));
  });
});

describe('gear quantisation', () => {
  it('returns gears 1..5 for any speed', () => {
    for (let speed = -5; speed <= 120; speed += 0.5) {
      const gear = gearFor(speed);
      expect(gear).toBeGreaterThanOrEqual(1);
      expect(gear).toBeLessThanOrEqual(5);
      expect(Number.isInteger(gear)).toBe(true);
    }
  });

  it('shifts up as speed increases and reaches fifth', () => {
    let last = 1;
    for (let speed = 0; speed <= 80; speed += 1) {
      const gear = gearFor(speed);
      expect(gear).toBeGreaterThanOrEqual(last);
      last = gear;
    }
    expect(gearFor(0)).toBe(1);
    expect(gearFor(80)).toBe(5);
  });

  it('keeps the in-gear fraction in 0..1 and resets at each shift', () => {
    const speeds = AudioConfig.engine.gearSpeeds;
    for (let speed = 0; speed <= 80; speed += 0.25) {
      const f = gearFraction(speed);
      expect(f).toBeGreaterThanOrEqual(0);
      expect(f).toBeLessThanOrEqual(1);
    }
    for (let i = 1; i < speeds.length; i++) expect(gearFraction(speeds[i])).toBeCloseTo(0);
  });

  it('dips the effective rpm briefly on a gear change', () => {
    const model = new EngineModel();
    const before = model.effectiveRpm(0.8, 8, 0);
    const atShift = model.effectiveRpm(0.8, 10, 0.1);
    const after = model.effectiveRpm(0.8, 10, 0.1 + AudioConfig.engine.shiftSeconds + 0.05);
    expect(model.gear).toBe(2);
    expect(atShift).toBeLessThan(after);
    expect(atShift).toBeLessThan(before);
    expect(after).toBeGreaterThan(0);
    expect(after).toBeLessThanOrEqual(1);
  });

  it('follows rpm01 when standing still', () => {
    const model = new EngineModel();
    const idle = model.effectiveRpm(0, 0, 10);
    const revving = model.effectiveRpm(1, 0, 11);
    expect(idle).toBeCloseTo(0);
    expect(revving).toBeCloseTo(1);
  });
});
