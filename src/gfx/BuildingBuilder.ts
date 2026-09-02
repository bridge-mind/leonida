import { Config } from '../core/Config';
import type { Random } from '../core/Random';
import type { BuildingStyle, DistrictDef } from '../data/districts';
import { Groups, type Physics } from '../physics/Physics';
import type { BatchLayer, ChunkBatches } from '../world/city/ChunkBatches';
import type { StaticEntities } from '../world/city/StaticEntities';
import { NO_BOTTOM, SIDES_ONLY } from './GeometryBatch';

export type Facing = 'N' | 'S' | 'E' | 'W';

/** Inner lot rectangle; `facing` is the side on which the street lies. */
export interface Lot {
  x0: number;
  z0: number;
  x1: number;
  z1: number;
  facing: Facing;
  district: DistrictDef;
}

export interface BuildingRecord {
  x: number;
  z: number;
  w: number;
  d: number;
  h: number;
  style: BuildingStyle;
  facing: Facing;
}

export interface BuildContext {
  batches: ChunkBatches;
  physics: Physics;
  entities: StaticEntities;
}

export const NEON_COLORS = [0xff4fa3, 0x35e6ff, 0xff2d95, 0xb6ff3a, 0xffb347, 0xa35bff, 0x3affd0] as const;
const FLOOR = Config.world.floorHeight;
const WINDOW_UV = 12.8;
const LOT_Y = Config.world.lotSurfaceY;

export const FACING_NORMAL: Record<Facing, { x: number; z: number }> = {
  N: { x: 0, z: -1 },
  S: { x: 0, z: 1 },
  E: { x: 1, z: 0 },
  W: { x: -1, z: 0 },
};

function shade(color: number, factor: number): number {
  const r = Math.min(255, Math.round(((color >> 16) & 255) * factor));
  const g = Math.min(255, Math.round(((color >> 8) & 255) * factor));
  const b = Math.min(255, Math.round((color & 255) * factor));
  return (r << 16) | (g << 8) | b;
}

/**
 * Builds one stylized building into the chunk batches and registers its collider(s).
 * Returns null when the lot is too small for the style.
 */
export function addBuilding(ctx: BuildContext, lot: Lot, style: BuildingStyle, rng: Random): BuildingRecord | null {
  const lotW = lot.x1 - lot.x0;
  const lotD = lot.z1 - lot.z0;
  if (lotW < 6 || lotD < 6) return null;
  switch (style) {
    case 'artDeco':
      return artDeco(ctx, lot, rng);
    case 'highRise':
      return highRise(ctx, lot, rng);
    case 'mixedUse':
      return mixedUse(ctx, lot, rng);
    case 'suburban':
      return suburban(ctx, lot, rng);
    case 'industrial':
      return industrial(ctx, lot, rng);
    case 'swamp':
      return shack(ctx, lot, rng);
  }
}

/** Solid box with a collider; UV offset staggers the lit-window pattern per building. */
export function solidBox(
  ctx: BuildContext,
  layer: BatchLayer,
  cx: number,
  y0: number,
  cz: number,
  w: number,
  h: number,
  d: number,
  color: number,
  rng: Random,
  opts?: { uvScale?: number; topColor?: number; collide?: boolean; glass?: boolean; sidesOnly?: boolean },
): void {
  const batch = ctx.batches.layer(cx, cz, layer);
  batch.addBox(cx, y0 + h / 2, cz, w, h, d, color, {
    uvScale: opts?.uvScale ?? WINDOW_UV,
    uvOffset: rng.int(0, 3) * 0.25,
    topColor: opts?.topColor ?? shade(color, 0.55),
    faces: opts?.sidesOnly ? SIDES_ONLY : NO_BOTTOM,
  });
  if (opts?.collide !== false) {
    ctx.physics.addStaticCuboid(
      { x: cx, y: y0 + h / 2, z: cz },
      { x: w / 2, y: h / 2, z: d / 2 },
      { entity: opts?.glass ? ctx.entities.glassBuilding : ctx.entities.building, groups: Groups.STATIC | Groups.BULLET_BLOCKER },
    );
  }
}

/** Thin decorative box attached to a face; `along` offsets it along the face from its center. */
export function faceBox(
  ctx: BuildContext,
  layer: BatchLayer,
  cx: number,
  cz: number,
  hw: number,
  hd: number,
  facing: Facing,
  y: number,
  alongSize: number,
  height: number,
  thickness: number,
  color: number,
  along = 0,
  protrude = 0,
): void {
  const n = FACING_NORMAL[facing];
  const fx = cx + n.x * hw;
  const fz = cz + n.z * hd;
  const tx = -n.z;
  const tz = n.x;
  const bx = fx + n.x * (thickness / 2 + protrude) + tx * along;
  const bz = fz + n.z * (thickness / 2 + protrude) + tz * along;
  const sx = n.x === 0 ? alongSize : thickness;
  const sz = n.x === 0 ? thickness : alongSize;
  ctx.batches.layer(bx, bz, layer).addBox(bx, y, bz, sx, height, sz, color, { uvScale: 4 });
}

function footprint(lot: Lot, insetStreet: number, insetOther: number): { cx: number; cz: number; w: number; d: number } {
  let x0 = lot.x0 + insetOther;
  let x1 = lot.x1 - insetOther;
  let z0 = lot.z0 + insetOther;
  let z1 = lot.z1 - insetOther;
  if (lot.facing === 'N') z0 = lot.z0 + insetStreet;
  if (lot.facing === 'S') z1 = lot.z1 - insetStreet;
  if (lot.facing === 'W') x0 = lot.x0 + insetStreet;
  if (lot.facing === 'E') x1 = lot.x1 - insetStreet;
  return { cx: (x0 + x1) / 2, cz: (z0 + z1) / 2, w: x1 - x0, d: z1 - z0 };
}

function artDeco(ctx: BuildContext, lot: Lot, rng: Random): BuildingRecord {
  const { cx, cz, w, d } = footprint(lot, 0.4, rng.range(0.8, 2.0));
  const floors = rng.int(lot.district.floors[0], lot.district.floors[1]);
  const h = floors * FLOOR;
  const color = rng.pick(lot.district.palette);
  const accent = rng.pick(lot.district.palette);
  const neon = rng.pick(NEON_COLORS);
  solidBox(ctx, 'deco', cx, LOT_Y, cz, w, h, d, color, rng);
  // Base band and roof parapet in the accent tone.
  solidBox(ctx, 'flat', cx, LOT_Y, cz, w + 0.3, 0.7, d + 0.3, shade(accent, 0.9), rng, { collide: false, topColor: shade(accent, 0.9) });
  solidBox(ctx, 'flat', cx, LOT_Y + h, cz, w + 0.2, 0.5, d + 0.2, 0xfff6e6, rng, { collide: false, topColor: 0xfff6e6, sidesOnly: true });
  const hw = w / 2;
  const hd = d / 2;
  // Stepped upper tier and a central fin with a vertical neon strip.
  const stepped = rng.chance(0.55);
  if (stepped) solidBox(ctx, 'deco', cx, LOT_Y + h, cz, w * 0.62, FLOOR, d * 0.62, shade(color, 1.05), rng, { collide: false });
  const finH = h * 0.3 + 3.5;
  const finY = LOT_Y + h + 3.5 - finH / 2;
  faceBox(ctx, 'flat', cx, cz, hw, hd, lot.facing, finY, 1.6, finH, 0.8, accent, 0, -0.2);
  faceBox(ctx, 'neon', cx, cz, hw, hd, lot.facing, finY + 0.3, 0.18, finH * 0.7, 0.18, neon, 0, 0.62);
  // Horizontal neon tubes along the street-facing roof edge and above the entrance.
  const along = Math.min(w, d) * 0.9;
  const faceLen = lot.facing === 'N' || lot.facing === 'S' ? w : d;
  faceBox(ctx, 'neon', cx, cz, hw, hd, lot.facing, LOT_Y + h - 0.35, faceLen * 0.9, 0.14, 0.14, neon, 0, 0.05);
  faceBox(ctx, 'neon', cx, cz, hw, hd, lot.facing, LOT_Y + FLOOR + 0.3, Math.min(faceLen * 0.5, along), 0.12, 0.12, rng.pick(NEON_COLORS), 0, 0.3);
  // Awning over the entrance and vertical pilasters.
  faceBox(ctx, 'flat', cx, cz, hw, hd, lot.facing, LOT_Y + FLOOR + 0.1, Math.min(faceLen * 0.5, along), 0.12, 1.6, shade(accent, 0.95), 0, 0.8);
  for (const s of [-1, 1]) faceBox(ctx, 'flat', cx, cz, hw, hd, lot.facing, LOT_Y + h / 2, 0.5, h, 0.3, 0xfff6e6, (s * faceLen) / 2 - s * 0.6, 0);
  return { x: cx, z: cz, w, d, h: h + (stepped ? FLOOR : 0), style: 'artDeco', facing: lot.facing };
}

function highRise(ctx: BuildContext, lot: Lot, rng: Random): BuildingRecord {
  const { cx, cz, w, d } = footprint(lot, 0.3, 0.6);
  const podiumH = 2 * FLOOR;
  const podiumColor = rng.pick([0x8a8d93, 0x9aa0a8, 0x777b82, 0xa8a49c]);
  solidBox(ctx, 'walls', cx, LOT_Y, cz, w, podiumH, d, podiumColor, rng);
  // Tower: footprint 55–80% of the podium, taller toward the downtown core for a skyline peak.
  const tw = w * rng.range(0.55, 0.8);
  const td = d * rng.range(0.55, 0.8);
  const tx = cx + rng.range(-(w - tw) / 2, (w - tw) / 2);
  const tz = cz + rng.range(-(d - td) / 2, (d - td) / 2);
  const core = 1 - Math.min(1, Math.hypot(cx - 60, cz + 60) / 380);
  const floors = Math.round(rng.range(lot.district.floors[0], lot.district.floors[1]) * (0.55 + 0.75 * core));
  const towerH = Math.max(6, floors) * FLOOR;
  const color = rng.pick(lot.district.palette);
  const setback = rng.chance(0.35) && towerH > 50;
  const mainH = setback ? towerH * 0.72 : towerH;
  solidBox(ctx, 'glass', tx, LOT_Y + podiumH, tz, tw, mainH, td, color, rng, { glass: true, topColor: 0x2a2f38 });
  if (setback) solidBox(ctx, 'glass', tx, LOT_Y + podiumH + mainH, tz, tw * 0.7, towerH - mainH, td * 0.7, shade(color, 1.08), rng, { glass: true, topColor: 0x2a2f38 });
  const roofY = LOT_Y + podiumH + towerH;
  const roofScale = setback ? 0.7 : 1;
  solidBox(ctx, 'flat', tx, roofY, tz, tw * roofScale * 0.3, 2.2, td * roofScale * 0.3, 0x4a4e55, rng, { collide: false, topColor: 0x4a4e55 });
  if (towerH > 75) {
    const antennaH = rng.range(6, 14);
    solidBox(ctx, 'flat', tx, roofY + 2.2, tz, 0.4, antennaH, 0.4, 0xbfc4cc, rng, { collide: false, topColor: 0xff3030 });
  }
  // Entrance canopy on the street face.
  const faceLen = lot.facing === 'N' || lot.facing === 'S' ? w : d;
  faceBox(ctx, 'flat', cx, cz, w / 2, d / 2, lot.facing, LOT_Y + 4.2, Math.min(faceLen * 0.4, 12), 0.25, 2.5, 0x2a2f38, 0, 1.25);
  return { x: cx, z: cz, w, d, h: podiumH + towerH, style: 'highRise', facing: lot.facing };
}

function mixedUse(ctx: BuildContext, lot: Lot, rng: Random): BuildingRecord {
  const { cx, cz, w, d } = footprint(lot, 0.3, rng.range(0.4, 1.2));
  const floors = rng.int(lot.district.floors[0], lot.district.floors[1]);
  const h = floors * FLOOR;
  const color = rng.pick(lot.district.palette);
  solidBox(ctx, 'walls', cx, LOT_Y, cz, w, h, d, color, rng);
  solidBox(ctx, 'flat', cx, LOT_Y + h, cz, w + 0.2, 0.6, d + 0.2, shade(color, 0.8), rng, { collide: false, topColor: shade(color, 0.8), sidesOnly: true });
  const faceLen = lot.facing === 'N' || lot.facing === 'S' ? w : d;
  const awning = rng.pick([0xd94a4a, 0x3a7a3a, 0xf0c050, 0xffffff, 0x3a5fa0]);
  faceBox(ctx, 'flat', cx, cz, w / 2, d / 2, lot.facing, LOT_Y + FLOOR + 0.15, faceLen * 0.85, 0.14, 1.3, awning, 0, 0.65);
  faceBox(ctx, 'flat', cx, cz, w / 2, d / 2, lot.facing, LOT_Y + 1.1, faceLen * 0.5, 2.2, 0.08, 0x2a3a4a, 0, 0);
  if (rng.chance(0.3) && floors > 1) {
    // Balcony railing on the upper floor.
    faceBox(ctx, 'flat', cx, cz, w / 2, d / 2, lot.facing, LOT_Y + FLOOR * (floors - 1) + 0.5, faceLen * 0.6, 1.0, 0.1, 0x3a3a3a, 0, 1.2);
    faceBox(ctx, 'flat', cx, cz, w / 2, d / 2, lot.facing, LOT_Y + FLOOR * (floors - 1) + 0.05, faceLen * 0.6, 0.12, 1.3, shade(color, 0.85), 0, 0);
  }
  if (rng.chance(0.4)) {
    // Mural: colored panels on a side wall.
    const side: Facing = lot.facing === 'N' || lot.facing === 'S' ? (rng.chance(0.5) ? 'E' : 'W') : rng.chance(0.5) ? 'N' : 'S';
    const sideLen = side === 'N' || side === 'S' ? w : d;
    const panels = rng.int(4, 7);
    for (let i = 0; i < panels; i++) {
      const pw = rng.range(1.5, sideLen * 0.35);
      const ph = rng.range(1.5, h * 0.6);
      const along = rng.range(-sideLen / 2 + pw / 2, sideLen / 2 - pw / 2);
      const y = LOT_Y + rng.range(ph / 2 + 0.3, h - ph / 2 - 0.3);
      faceBox(ctx, 'flat', cx, cz, w / 2, d / 2, side, y, pw, ph, 0.05, rng.pick([0xff4fa3, 0x35e6ff, 0xffd23f, 0x4ee06a, 0xff7a3a, 0xa35bff, 0xffffff]), along, 0);
    }
  }
  if (rng.chance(0.25)) solidBox(ctx, 'flat', cx + rng.range(-w / 4, w / 4), LOT_Y + h + 0.6, cz + rng.range(-d / 4, d / 4), 1.8, 2.2, 1.8, 0x6b5a4a, rng, { collide: false, topColor: 0x6b5a4a });
  return { x: cx, z: cz, w, d, h, style: 'mixedUse', facing: lot.facing };
}

function suburban(ctx: BuildContext, lot: Lot, rng: Random): BuildingRecord {
  const lotW = lot.x1 - lot.x0;
  const lotD = lot.z1 - lot.z0;
  const w = Math.min(lotW - 3, rng.range(9, 13));
  const d = Math.min(lotD - 4, rng.range(8, 11));
  const n = FACING_NORMAL[lot.facing];
  // House sits toward the back of the lot; a driveway runs to the street.
  const cx = (lot.x0 + lot.x1) / 2 - n.x * (lotW / 2 - w / 2 - 3.5);
  const cz = (lot.z0 + lot.z1) / 2 - n.z * (lotD / 2 - d / 2 - 3.5);
  const floors = rng.chance(0.4) ? 2 : 1;
  const h = floors * FLOOR;
  const color = rng.pick(lot.district.palette);
  solidBox(ctx, 'walls', cx, LOT_Y, cz, w, h, d, color, rng, { uvScale: 9.6 });
  const roofColor = rng.pick([0x7a3b2e, 0x4a4a4a, 0x6b5a4a, 0x8a3a3a, 0x3f4f5f]);
  const alongZ = lot.facing === 'E' || lot.facing === 'W';
  ctx.batches.layer(cx, cz, 'flat').addGableRoof(cx, LOT_Y + h, cz, w + 0.8, d + 0.8, 2.2, roofColor, alongZ);
  // Porch step and driveway slab.
  faceBox(ctx, 'flat', cx, cz, w / 2, d / 2, lot.facing, LOT_Y + 0.15, 3, 0.3, 1.6, 0xd8d0c0, 0, 0);
  const driveLen = alongZ ? lotW / 2 - w / 2 : lotD / 2 - d / 2;
  const dcx = cx + n.x * (w / 2 + driveLen / 2 + 0.5);
  const dcz = cz + n.z * (d / 2 + driveLen / 2 + 0.5);
  ctx.batches.layer(dcx, dcz, 'concrete').addBox(dcx, LOT_Y + 0.02, dcz, alongZ ? driveLen + 1 : 3, 0.03, alongZ ? 3 : driveLen + 1, 0xb8b4ac, { uvScale: 8, faces: { front: false, back: false, left: false, right: false, top: true, bottom: false } });
  return { x: cx, z: cz, w, d, h: h + 2.2, style: 'suburban', facing: lot.facing };
}

function industrial(ctx: BuildContext, lot: Lot, rng: Random): BuildingRecord {
  const { cx, cz, w, d } = footprint(lot, 1.0, rng.range(1.5, 3));
  const h = rng.range(8, 12);
  const color = rng.pick(lot.district.palette);
  solidBox(ctx, 'walls', cx, LOT_Y, cz, w, h, d, color, rng, { uvScale: 25.6, topColor: shade(color, 0.5) });
  solidBox(ctx, 'flat', cx, LOT_Y + h, cz, w + 0.4, 0.5, d + 0.4, shade(color, 0.6), rng, { collide: false, topColor: shade(color, 0.6), sidesOnly: true });
  for (let i = 0; i < 3; i++) {
    solidBox(ctx, 'flat', cx + rng.range(-w / 3, w / 3), LOT_Y + h, cz + rng.range(-d / 3, d / 3), 1.6, 1.2, 1.6, 0x9a9a9a, rng, { collide: false, topColor: 0x9a9a9a });
  }
  const faceLen = lot.facing === 'N' || lot.facing === 'S' ? w : d;
  // Big roller door and a rust stripe.
  faceBox(ctx, 'flat', cx, cz, w / 2, d / 2, lot.facing, LOT_Y + 2.6, Math.min(7, faceLen * 0.35), 5.2, 0.06, 0x3a3f44, -faceLen * 0.2, 0);
  faceBox(ctx, 'flat', cx, cz, w / 2, d / 2, lot.facing, LOT_Y + h - 1.2, faceLen * 0.96, 0.6, 0.05, shade(color, 0.7), 0, 0);
  if (rng.chance(0.6)) {
    // Office annex beside the entrance.
    const n = FACING_NORMAL[lot.facing];
    const ax = cx + n.x * (w / 2 + 3) + -n.z * (faceLen * 0.3);
    const az = cz + n.z * (d / 2 + 3) + n.x * (faceLen * 0.3);
    solidBox(ctx, 'walls', ax, LOT_Y, az, 7, 3.6, 6, shade(color, 1.15), rng, { uvScale: 9.6 });
  }
  return { x: cx, z: cz, w, d, h, style: 'industrial', facing: lot.facing };
}

function shack(ctx: BuildContext, lot: Lot, rng: Random): BuildingRecord {
  const { cx, cz, w, d } = footprint(lot, 0.5, 0.5);
  const color = rng.pick(lot.district.palette);
  // Stilts, a plank floor, walls and a tin roof.
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) solidBox(ctx, 'flat', cx + (sx * w) / 2.4, LOT_Y, cz + (sz * d) / 2.4, 0.3, 1.2, 0.3, 0x4a3a2a, rng, { collide: false });
  solidBox(ctx, 'planks', cx, LOT_Y + 1.2, cz, w + 1.5, 0.2, d + 1.5, 0xa57a4e, rng, { uvScale: 4, topColor: 0xa57a4e });
  solidBox(ctx, 'planks', cx, LOT_Y + 1.4, cz, w, 3.0, d, color, rng, { uvScale: 4, topColor: color });
  ctx.batches.layer(cx, cz, 'flat').addGableRoof(cx, LOT_Y + 4.4, cz, w + 1.2, d + 1.2, 1.2, 0x7a7f86, false);
  return { x: cx, z: cz, w, d, h: 5.6, style: 'swamp', facing: lot.facing };
}
