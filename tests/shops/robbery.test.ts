import { describe, expect, it } from 'vitest';
import { RobberyClock } from '../../src/systems/robbery/progress';

const make = (): RobberyClock => new RobberyClock(12, 0.4, 2);

describe('RobberyClock', () => {
  it('fills the bag in exactly the configured duration', () => {
    const c = make();
    for (let i = 0; i < 11; i++) expect(c.tick(1)).toBe(false);
    expect(c.fraction).toBeCloseTo(11 / 12);
    expect(c.tick(1)).toBe(true);
    expect(c.fraction).toBe(1);
    expect(c.secondsLeft).toBe(0);
  });

  it('a gunshot cuts the remaining time by 40%', () => {
    const c = make();
    c.tick(2);
    expect(c.onShot()).toBe(true);
    expect(c.secondsLeft).toBeCloseTo(6);
    expect(c.shots).toBe(1);
  });

  it('only the first two shots count', () => {
    const c = make();
    expect(c.onShot()).toBe(true);
    expect(c.onShot()).toBe(true);
    expect(c.onShot()).toBe(false);
    expect(c.shots).toBe(2);
    expect(c.secondsLeft).toBeCloseTo(12 * 0.6 * 0.6);
  });

  it('progress fraction stays monotonic and reaches 1 after shots', () => {
    const c = make();
    let last = 0;
    c.tick(3);
    c.onShot();
    for (let i = 0; i < 20 && !c.done; i++) {
      c.tick(0.5);
      expect(c.fraction).toBeGreaterThanOrEqual(last);
      last = c.fraction;
    }
    expect(c.done).toBe(true);
    expect(c.fraction).toBe(1);
  });

  it('ignores non-positive deltas', () => {
    const c = make();
    c.tick(0);
    c.tick(-5);
    expect(c.secondsLeft).toBe(12);
    expect(c.fraction).toBe(0);
  });
});
