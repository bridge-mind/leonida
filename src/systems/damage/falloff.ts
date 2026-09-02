/**
 * Pure damage math (unit-tested in tests/weapons).
 */

/** Linear explosion falloff: 1 inside `radius * fullFraction`, 0 at and beyond `radius`. */
export function explosionFalloff(distance: number, radius: number, fullFraction: number): number {
  const full = radius * fullFraction;
  if (distance <= full) return 1;
  if (distance >= radius) return 0;
  return 1 - (distance - full) / (radius - full);
}

/** Damage after the body-part multiplier. */
export function partMultiplier(part: string | undefined, headshotMultiplier: number, headshotApplies: boolean): number {
  return part === 'head' && headshotApplies ? headshotMultiplier : 1;
}
