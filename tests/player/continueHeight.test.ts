import { describe, expect, it } from 'vitest';
import { CONTINUE_PROBE, continueFeetY } from '../../src/systems/player/index';

describe('Continue height', () => {
  it('keeps a saved interior height when the short probe misses', () => {
    expect(continueFeetY(600, null)).toBe(600);
  });

  it('stands on a floor the short probe actually hit', () => {
    expect(continueFeetY(600.4, 600)).toBeCloseTo(600.05);
  });

  it('keeps a jump instead of a non-finite hit', () => {
    expect(continueFeetY(12, Number.NaN)).toBe(12);
  });

  it('starts the probe just above the saved feet and only looks a few metres down', () => {
    expect(CONTINUE_PROBE.above).toBeLessThan(5);
    expect(CONTINUE_PROBE.drop).toBeLessThan(20);
    expect(600 + CONTINUE_PROBE.above - CONTINUE_PROBE.drop).toBeGreaterThan(500);
  });
});
