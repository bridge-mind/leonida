import { Vector3 } from 'three';
import { Config } from '../../core/Config';
import type { Random } from '../../core/Random';
import type { DistrictDef } from '../../data/districts';
import { addBuilding, type BuildContext, type BuildingRecord, type Facing, type Lot } from '../../gfx/BuildingBuilder';
import type { CityMaterials } from '../../gfx/Materials';
import type { Physics } from '../../physics/Physics';
import type { Districts } from '../Districts';
import type { Landmarks } from '../Landmarks';
import type { Props } from '../Props';
import type { RoadGraph } from '../RoadGraph';
import { ROAD_KINDS, ROAD_LINES, type RoadKind } from '../roadNetwork';
import type { ChunkBatches } from './ChunkBatches';
import type { StaticEntities } from './StaticEntities';

const SW = Config.world.sidewalkWidth;
const LOT_Y = Config.world.lotSurfaceY;

export interface Block {
  /** Inner rectangle (inside the sidewalks). */
  x0: number;
  z0: number;
  x1: number;
  z1: number;
  district: DistrictDef;
  /** Road kind on each side, null when that side has no road. */
  sides: Record<Facing, RoadKind | null>;
}

export interface BlocksOptions {
  graph: RoadGraph;
  batches: ChunkBatches;
  districts: Districts;
  landmarks: Landmarks;
  physics: Physics;
  entities: StaticEntities;
  rng: Random;
  props: Props;
  materials: CityMaterials;
}

/** Blocks that must not receive regular buildings (special areas fill them). */
export const RESERVED_BLOCKS: { x0: number; z0: number; x1: number; z1: number; reason: string }[] = [
  { x0: -600, z0: -660, x1: -400, z1: -530, reason: 'trailer park' },
  { x0: -700, z0: 750, x1: -400, z1: 850, reason: 'container yard' },
];

/** Does a road of any kind run along `axis` at `at` covering [from, to]? Returns its kind. */
export function roadAlong(axis: 'x' | 'z', at: number, from: number, to: number): RoadKind | null {
  for (const line of ROAD_LINES) {
    if (line.axis !== axis || Math.abs(line.at - at) > 0.01) continue;
    const lo = Math.min(line.from, line.to);
    const hi = Math.max(line.from, line.to);
    if (lo <= from + 0.01 && hi >= to - 0.01) return line.kind;
  }
  return null;
}

/** Enumerates every grid cell whose four sides are roads. */
export function findBlocks(districts: Districts): Block[] {
  const xs = [...new Set(ROAD_LINES.filter((l) => l.axis === 'z').map((l) => l.at))].sort((a, b) => a - b);
  const zs = [...new Set(ROAD_LINES.filter((l) => l.axis === 'x').map((l) => l.at))].sort((a, b) => a - b);
  const blocks: Block[] = [];
  for (let i = 0; i + 1 < xs.length; i++) {
    for (let j = 0; j + 1 < zs.length; j++) {
      const x0 = xs[i];
      const x1 = xs[i + 1];
      const z0 = zs[j];
      const z1 = zs[j + 1];
      const n = roadAlong('x', z0, x0, x1);
      const s = roadAlong('x', z1, x0, x1);
      const w = roadAlong('z', x0, z0, z1);
      const e = roadAlong('z', x1, z0, z1);
      if (!n || !s || !w || !e) continue;
      const cx = (x0 + x1) / 2;
      const cz = (z0 + z1) / 2;
      const district = districts.at(cx, cz) ?? districts.nearest(cx, cz);
      const inset = (k: RoadKind): number => ROAD_KINDS[k].width / 2 + (ROAD_KINDS[k].sidewalks ? SW : 2);
      blocks.push({
        x0: x0 + inset(w),
        x1: x1 - inset(e),
        z0: z0 + inset(n),
        z1: z1 - inset(s),
        district,
        sides: { N: n, S: s, W: w, E: e },
      });
    }
  }
  return blocks;
}

function isReserved(b: Block): boolean {
  const cx = (b.x0 + b.x1) / 2;
  const cz = (b.z0 + b.z1) / 2;
  return RESERVED_BLOCKS.some((r) => cx > r.x0 && cx < r.x1 && cz > r.z0 && cz < r.z1);
}

function faceable(kind: RoadKind | null): boolean {
  return kind !== null && kind !== 'highway' && kind !== 'dirt';
}

/**
 * Fills every block with a slab, perimeter lots facing the streets, buildings by
 * district style, parking lots and yard props. Returns the building count.
 */
export function fillBlocks(opts: BlocksOptions): { buildings: number; records: BuildingRecord[]; blocks: Block[] } {
  const { batches, districts, landmarks, physics, entities, rng, props, graph } = opts;
  const ctx: BuildContext = { batches, physics, entities };
  const blocks = findBlocks(districts);
  const records: BuildingRecord[] = [];

  for (const block of blocks) {
    if (block.district.style === 'swamp') continue;
    const w = block.x1 - block.x0;
    const d = block.z1 - block.z0;
    if (w < 8 || d < 8) continue;
    addSlab(block);
    if (isReserved(block)) continue;
    if (block.district.style === 'highRise') fillTowers(block);
    else fillPerimeter(block);
  }
  return { buildings: records.length, records, blocks };

  function addSlab(block: Block): void {
    const cx = (block.x0 + block.x1) / 2;
    const cz = (block.z0 + block.z1) / 2;
    const layer = block.district.style === 'suburban' || block.district.style === 'swamp' ? 'flat' : 'concrete';
    batches.layer(cx, cz, layer).addQuad(block.x0, block.z0, block.x1, block.z1, LOT_Y, block.district.lotColor, 8);
  }

  function overlapsLandmark(x0: number, z0: number, x1: number, z1: number): boolean {
    for (const l of landmarks.list) {
      const hw = l.footprint.w / 2 + 3;
      const hd = l.footprint.d / 2 + 3;
      if (x1 < l.position.x - hw || x0 > l.position.x + hw || z1 < l.position.z - hd || z0 > l.position.z + hd) continue;
      return true;
    }
    return false;
  }

  function placeLot(lot: Lot): void {
    if (overlapsLandmark(lot.x0, lot.z0, lot.x1, lot.z1)) return;
    if (rng.chance(lot.district.emptyLotChance)) {
      emptyLot(lot);
      return;
    }
    const rec = addBuilding(ctx, lot, lot.district.style, rng);
    if (rec) records.push(rec);
  }

  /** Parking lot with spots for Agent 2, or a yard with a dumpster. */
  function emptyLot(lot: Lot): void {
    const w = lot.x1 - lot.x0;
    const d = lot.z1 - lot.z0;
    const cx = (lot.x0 + lot.x1) / 2;
    const cz = (lot.z0 + lot.z1) / 2;
    if (lot.district.style === 'suburban') {
      props.add('cypress', cx + rng.range(-w / 4, w / 4), LOT_Y, cz + rng.range(-d / 4, d / 4), rng.range(0, 6.28), rng.range(0.6, 0.9));
      return;
    }
    if (lot.district.style === 'industrial') {
      stackContainers(lot);
      return;
    }
    batches.layer(cx, cz, 'concrete').addQuad(lot.x0 + 0.5, lot.z0 + 0.5, lot.x1 - 0.5, lot.z1 - 0.5, LOT_Y + 0.01, 0x7d7d7f, 8);
    const alongZ = lot.facing === 'E' || lot.facing === 'W';
    const rows = Math.floor((alongZ ? d : w) / 3.2);
    const heading = alongZ ? Math.PI / 2 : 0;
    for (let i = 0; i < rows; i++) {
      const t = -((rows - 1) / 2) + i;
      const x = alongZ ? cx : cx + t * 3.2;
      const z = alongZ ? cz + t * 3.2 : cz;
      graph.parkingSpots.push({ position: new Vector3(x, LOT_Y, z), heading, edgeId: -1, side: 1, kind: 'lot' });
    }
    props.add('dumpster', lot.x0 + 1.5, LOT_Y, lot.z0 + 1.5, rng.range(0, 6.28));
    if (rng.chance(0.5)) props.add('bollard', lot.x1 - 1, LOT_Y, lot.z1 - 1);
  }

  function stackContainers(lot: Lot): void {
    const w = lot.x1 - lot.x0;
    const d = lot.z1 - lot.z0;
    const alongX = w >= d;
    const rows = Math.max(1, Math.floor((alongX ? d : w) / 3.6));
    const perRow = Math.max(1, Math.floor((alongX ? w : d) / 13.5));
    const colors = [0x9c3b2b, 0x2f5f9c, 0x3a7a4a, 0xd9a52c, 0xd0d0d0, 0x5a5a5a];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < perRow; c++) {
        if (rng.chance(0.25)) continue;
        const u = lot.x0 + (alongX ? 6.8 + c * 13.5 : 1.8 + r * 3.6);
        const v = lot.z0 + (alongX ? 1.8 + r * 3.6 : 6.8 + c * 13.5);
        const stack = rng.int(1, 3);
        for (let s = 0; s < stack; s++) props.add('container', u, LOT_Y + s * 2.6, v, alongX ? 0 : Math.PI / 2, 1, rng.pick(colors));
      }
    }
  }

  function fillPerimeter(block: Block): void {
    const w = block.x1 - block.x0;
    const d = block.z1 - block.z0;
    const dist = block.district;
    const depth = Math.min(dist.lotDepth, Math.floor(Math.min(w, d) / 2) - 1);
    if (depth < 8) return;
    // N/S sides span the full width; E/W sides take the remaining middle band.
    const sides: { facing: Facing; kind: RoadKind | null }[] = [
      { facing: 'N', kind: block.sides.N },
      { facing: 'S', kind: block.sides.S },
      { facing: 'W', kind: block.sides.W },
      { facing: 'E', kind: block.sides.E },
    ];
    for (const side of sides) {
      if (!faceable(side.kind)) continue;
      const horizontal = side.facing === 'N' || side.facing === 'S';
      const from = horizontal ? block.x0 : block.z0 + (faceable(block.sides.N) ? depth : 0);
      const to = horizontal ? block.x1 : block.z1 - (faceable(block.sides.S) ? depth : 0);
      if (to - from < dist.lotWidth[0]) continue;
      let cursor = from;
      while (to - cursor >= dist.lotWidth[0] * 0.8) {
        let lw = rng.range(dist.lotWidth[0], dist.lotWidth[1]);
        if (to - cursor - lw < dist.lotWidth[0] * 0.8) lw = to - cursor;
        const lot: Lot = horizontal
          ? {
              x0: cursor,
              x1: cursor + lw,
              z0: side.facing === 'N' ? block.z0 : block.z1 - depth,
              z1: side.facing === 'N' ? block.z0 + depth : block.z1,
              facing: side.facing,
              district: dist,
            }
          : {
              z0: cursor,
              z1: cursor + lw,
              x0: side.facing === 'W' ? block.x0 : block.x1 - depth,
              x1: side.facing === 'W' ? block.x0 + depth : block.x1,
              facing: side.facing,
              district: dist,
            };
        placeLot(lot);
        cursor += lw;
      }
    }
    // Courtyard: benches and trees in the leftover middle for larger blocks.
    const innerW = w - 2 * depth;
    const innerD = d - 2 * depth;
    if (innerW > 12 && innerD > 12) {
      const cx = (block.x0 + block.x1) / 2;
      const cz = (block.z0 + block.z1) / 2;
      if (dist.style === 'suburban' || dist.style === 'mixedUse') {
        batches.layer(cx, cz, 'flat').addQuad(cx - innerW / 2, cz - innerD / 2, cx + innerW / 2, cz + innerD / 2, LOT_Y + 0.01, 0x6f8f4d, 8);
        const trees = Math.min(6, Math.floor((innerW * innerD) / 400));
        for (let i = 0; i < trees; i++) props.add('palm', cx + rng.range(-innerW / 2 + 2, innerW / 2 - 2), LOT_Y, cz + rng.range(-innerD / 2 + 2, innerD / 2 - 2), rng.range(0, 6.28), rng.range(0.8, 1.1));
      }
    }
  }

  function fillTowers(block: Block): void {
    const w = block.x1 - block.x0;
    const d = block.z1 - block.z0;
    const nx = w > 64 ? 2 : 1;
    const nz = d > 64 ? 2 : 1;
    const lw = w / nx;
    const ld = d / nz;
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < nz; j++) {
        const x0 = block.x0 + i * lw + 1.5;
        const x1 = block.x0 + (i + 1) * lw - 1.5;
        const z0 = block.z0 + j * ld + 1.5;
        const z1 = block.z0 + (j + 1) * ld - 1.5;
        // Face the nearest faceable street.
        let facing: Facing = 'S';
        if (nz === 2) facing = j === 0 ? 'N' : 'S';
        else if (nx === 2) facing = i === 0 ? 'W' : 'E';
        if (!faceable(block.sides[facing])) facing = (['N', 'S', 'E', 'W'] as Facing[]).find((f) => faceable(block.sides[f])) ?? 'S';
        placeLot({ x0, x1, z0, z1, facing, district: block.district });
      }
    }
  }
}
