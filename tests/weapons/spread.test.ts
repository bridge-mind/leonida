import { describe, expect, it } from 'vitest';
import { angleBetween, npcSpreadHalfAngle, randomInCone, spreadHalfAngle } from '../../src/systems/weapons/spread';

function seeded(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

describe('spread', () => {
  it('converts the full cone angle to a half angle in radians', () => {
    expect(spreadHalfAngle(0)).toBe(0);
    expect(spreadHalfAngle(8)).toBeCloseTo((4 * Math.PI) / 180, 10);
  });

  it('keeps every sample inside the cone and normalised', () => {
    const rand = seeded(7);
    const dirs = [
      { x: 0, y: 0, z: -1 },
      { x: 1, y: 0, z: 0 },
      { x: 0, y: 1, z: 0 },
      { x: 0, y: -1, z: 0 },
      { x: 1 / Math.sqrt(3), y: 1 / Math.sqrt(3), z: 1 / Math.sqrt(3) },
    ];
    const half = spreadHalfAngle(8);
    const out = { x: 0, y: 0, z: 0 };
    for (const dir of dirs) {
      for (let i = 0; i < 500; i++) {
        randomInCone(dir, half, out, rand);
        expect(Math.hypot(out.x, out.y, out.z)).toBeCloseTo(1, 6);
        expect(angleBetween(dir, out)).toBeLessThanOrEqual(half + 1e-6);
      }
    }
  });

  it('actually spreads (samples are not all identical) and covers the cone', () => {
    const rand = seeded(42);
    const dir = { x: 0, y: 0, z: -1 };
    const half = spreadHalfAngle(8);
    const out = { x: 0, y: 0, z: 0 };
    let maxAngle = 0;
    const seen = new Set<string>();
    for (let i = 0; i < 2000; i++) {
      randomInCone(dir, half, out, rand);
      maxAngle = Math.max(maxAngle, angleBetween(dir, out));
      seen.add(`${out.x.toFixed(4)},${out.y.toFixed(4)}`);
    }
    expect(seen.size).toBeGreaterThan(1500);
    expect(maxAngle).toBeGreaterThan(half * 0.95);
  });

  it('returns the exact direction for a zero cone', () => {
    const out = { x: 0, y: 0, z: 0 };
    randomInCone({ x: 0, y: 0, z: -1 }, 0, out, seeded(1));
    expect(out).toEqual({ x: 0, y: 0, z: -1 });
  });

  it('widens the NPC cone with inaccuracy and adds a miss offset', () => {
    const base = spreadHalfAngle(1.5);
    expect(npcSpreadHalfAngle(1.5, 1, 3, 0.5, 0.12, () => 0.99)).toBeCloseTo(base, 10);
    // accuracy 0 → spread × 4, no miss when the roll is high
    expect(npcSpreadHalfAngle(1.5, 0, 3, 0.5, 0.12, () => 0.99)).toBeCloseTo(base * 4, 10);
    // accuracy 0 → 50 % miss chance; a low roll adds the offset
    expect(npcSpreadHalfAngle(1.5, 0, 3, 0.5, 0.12, () => 0.1)).toBeCloseTo(base * 4 + 0.12, 10);
    // perfect accuracy never misses
    expect(npcSpreadHalfAngle(1.5, 1, 3, 0.5, 0.12, () => 0)).toBeCloseTo(base, 10);
  });
});
