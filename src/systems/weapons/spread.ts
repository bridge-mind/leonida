import type { V3 } from '../../core/entities';

/**
 * Pure spread math (unit-tested in tests/weapons). `dir` must be normalised; results are
 * written into `out` so hot paths do not allocate.
 */

/** Uniformly distributed direction inside a cone of `halfAngle` radians around `dir`. */
export function randomInCone(dir: V3, halfAngle: number, out: V3, rand: () => number = Math.random): V3 {
  if (halfAngle <= 0) {
    out.x = dir.x;
    out.y = dir.y;
    out.z = dir.z;
    return out;
  }
  // Uniform over the spherical cap: cos(theta) uniform in [cos(halfAngle), 1].
  const cosMax = Math.cos(halfAngle);
  const cosT = cosMax + (1 - cosMax) * rand();
  const sinT = Math.sqrt(Math.max(0, 1 - cosT * cosT));
  const phi = rand() * Math.PI * 2;
  const ox = sinT * Math.cos(phi);
  const oy = sinT * Math.sin(phi);

  // Orthonormal basis (u, v) perpendicular to dir.
  let ux: number;
  let uy: number;
  let uz: number;
  if (Math.abs(dir.y) < 0.99) {
    // u = normalize(dir × up)
    ux = dir.z;
    uy = 0;
    uz = -dir.x;
  } else {
    // dir is nearly vertical: use +X as the helper axis
    ux = 0;
    uy = dir.z;
    uz = -dir.y;
  }
  const ul = Math.hypot(ux, uy, uz) || 1;
  ux /= ul;
  uy /= ul;
  uz /= ul;
  // v = dir × u
  const vx = dir.y * uz - dir.z * uy;
  const vy = dir.z * ux - dir.x * uz;
  const vz = dir.x * uy - dir.y * ux;

  out.x = dir.x * cosT + ux * ox + vx * oy;
  out.y = dir.y * cosT + uy * ox + vy * oy;
  out.z = dir.z * cosT + uz * ox + vz * oy;
  const len = Math.hypot(out.x, out.y, out.z) || 1;
  out.x /= len;
  out.y /= len;
  out.z /= len;
  return out;
}

/** Full cone angle in degrees → half angle in radians. */
export function spreadHalfAngle(spreadDeg: number): number {
  return (spreadDeg * Math.PI) / 360;
}

/**
 * NPC cone half-angle: table spread widened by inaccuracy, plus a deliberate wide miss
 * with probability `(1 - accuracy) * missChance`.
 */
export function npcSpreadHalfAngle(
  spreadDeg: number,
  accuracy: number,
  penalty: number,
  missChance: number,
  missOffsetRad: number,
  rand: () => number = Math.random,
): number {
  const a = Math.min(1, Math.max(0, accuracy));
  let half = spreadHalfAngle(spreadDeg) * (1 + penalty * (1 - a));
  if (rand() < (1 - a) * missChance) half += missOffsetRad;
  return half;
}

export function angleBetween(a: V3, b: V3): number {
  const d = a.x * b.x + a.y * b.y + a.z * b.z;
  return Math.acos(Math.min(1, Math.max(-1, d)));
}
