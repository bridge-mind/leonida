import { Group, Mesh, Vector3 } from 'three';
import { Config } from '../../core/Config';
import type { Random } from '../../core/Random';
import type { LandmarkDef } from '../../data/landmarks';
import { FACING_NORMAL, faceBox, solidBox, type BuildContext, type Facing } from '../../gfx/BuildingBuilder';
import { GeometryBatch } from '../../gfx/GeometryBatch';
import type { CityMaterials } from '../../gfx/Materials';
import type { SignAtlas } from '../../gfx/Textures';
import type { Physics } from '../../physics/Physics';
import type { Landmarks } from '../Landmarks';
import type { Props } from '../Props';
import type { ChunkBatches } from './ChunkBatches';
import type { StaticEntities } from './StaticEntities';

const LOT_Y = Config.world.lotSurfaceY;
const FLOOR = Config.world.floorHeight;

export interface LandmarkBuildOptions {
  landmarks: Landmarks;
  batches: ChunkBatches;
  physics: Physics;
  entities: StaticEntities;
  signs: SignAtlas;
  materials: CityMaterials;
  parent: Group;
  props: Props;
  rng: Random;
}

/** The face of the building the door is on, derived from the door position. */
function doorFacing(l: LandmarkDef): Facing {
  const dx = l.door.x - l.position.x;
  const dz = l.door.z - l.position.z;
  const rx = Math.abs(dx) / (l.footprint.w / 2);
  const rz = Math.abs(dz) / (l.footprint.d / 2);
  if (rx > rz) return dx > 0 ? 'E' : 'W';
  return dz > 0 ? 'S' : 'N';
}

/**
 * Hand-styled landmark buildings with lit signs over their doors. Regular lots
 * that overlap a landmark footprint are skipped by the block filler.
 */
export function buildLandmarks(opts: LandmarkBuildOptions): { signMesh: Mesh } {
  const { landmarks, batches, physics, entities, signs, materials, parent, props, rng } = opts;
  const ctx: BuildContext = { batches, physics, entities };
  const signBatch = new GeometryBatch();

  for (const l of landmarks.list) {
    const facing = doorFacing(l);
    switch (l.style) {
      case 'casino':
        casino(l, facing);
        break;
      case 'prison':
        prison(l);
        break;
      case 'hospital':
        blockBuilding(l, facing, 0xf4f4f0, 'walls', 0xff3b3b);
        cross(l, facing);
        break;
      case 'police':
        blockBuilding(l, facing, 0x9fb0c8, 'walls', 0x3a6fd8);
        break;
      case 'bank':
        blockBuilding(l, facing, 0xd9cfb8, 'walls', 0xc9a24a);
        columns(l, facing);
        break;
      case 'jewelry':
        blockBuilding(l, facing, 0x2b2b33, 'walls', 0xff4fa3);
        break;
      case 'gunstore':
        blockBuilding(l, facing, 0x8a5a3a, 'flat', 0xff8a1f);
        break;
      case 'store':
        blockBuilding(l, facing, 0xf0e8d0, 'flat', 0x35e6ff);
        break;
      case 'gasStation':
        gasStation(l, facing);
        break;
      case 'diner':
        blockBuilding(l, facing, 0xffd23f, 'flat', 0xff5c5c);
        break;
      case 'apartment':
        blockBuilding(l, facing, 0xf2d38a, 'walls', 0x4ee06a);
        break;
      case 'respray':
        garage(l, facing, 0xa35bff);
        break;
      case 'chopShop':
        garage(l, facing, 0xffd23f);
        break;
      case 'marina':
        blockBuilding(l, facing, 0xe8f0f0, 'flat', 0x35e6ff);
        break;
      case 'club':
        blockBuilding(l, facing, 0x5a3a2a, 'planks', 0xff4fa3);
        break;
    }
    if (l.style !== 'prison') addSign(l, facing);
    addDoor(l, facing);
  }

  const signMesh = new Mesh(signBatch.build(), materials.signs);
  signMesh.name = 'landmarkSigns';
  signMesh.matrixAutoUpdate = false;
  parent.add(signMesh);
  return { signMesh };

  function blockBuilding(l: LandmarkDef, facing: Facing, color: number, layer: 'walls' | 'flat' | 'planks', trim: number): void {
    const { w, d, h } = l.footprint;
    solidBox(ctx, layer, l.position.x, LOT_Y, l.position.z, w, h, d, color, rng, { uvScale: layer === 'planks' ? 4 : 12.8 });
    solidBox(ctx, 'flat', l.position.x, LOT_Y + h, l.position.z, w + 0.3, 0.5, d + 0.3, trim, rng, { collide: false, topColor: trim, sidesOnly: true });
    const faceLen = facing === 'N' || facing === 'S' ? w : d;
    faceBox(ctx, 'flat', l.position.x, l.position.z, w / 2, d / 2, facing, LOT_Y + FLOOR + 0.2, faceLen * 0.9, 0.16, 1.4, trim, 0, 0.7);
    faceBox(ctx, 'neon', l.position.x, l.position.z, w / 2, d / 2, facing, LOT_Y + FLOOR + 0.35, faceLen * 0.85, 0.12, 0.12, trim, 0, 1.4);
  }

  function cross(l: LandmarkDef, facing: Facing): void {
    const { w, d, h } = l.footprint;
    faceBox(ctx, 'neon', l.position.x, l.position.z, w / 2, d / 2, facing, LOT_Y + h * 0.7, 1.2, 4.5, 0.2, 0xff3b3b, 0, 0.05);
    faceBox(ctx, 'neon', l.position.x, l.position.z, w / 2, d / 2, facing, LOT_Y + h * 0.7, 4.5, 1.2, 0.2, 0xff3b3b, 0, 0.05);
    // Helipad on the roof.
    solidBox(ctx, 'flat', l.position.x, LOT_Y + h + 0.5, l.position.z, 14, 0.3, 14, 0x3a3f46, rng, { collide: false, topColor: 0x3a3f46 });
    solidBox(ctx, 'flat', l.position.x, LOT_Y + h + 0.8, l.position.z, 6, 0.05, 1.2, 0xffffff, rng, { collide: false, topColor: 0xffffff });
  }

  function columns(l: LandmarkDef, facing: Facing): void {
    const { w, d, h } = l.footprint;
    const faceLen = facing === 'N' || facing === 'S' ? w : d;
    const count = Math.floor(faceLen / 5);
    for (let i = 0; i < count; i++) {
      const along = -faceLen / 2 + 2.5 + i * 5;
      faceBox(ctx, 'flat', l.position.x, l.position.z, w / 2, d / 2, facing, LOT_Y + h * 0.4, 1.0, h * 0.8, 1.0, 0xe8e0cc, along, 1.0);
    }
    faceBox(ctx, 'flat', l.position.x, l.position.z, w / 2, d / 2, facing, LOT_Y + h * 0.8 + 0.8, faceLen, 1.6, 2.6, 0xe8e0cc, 0, 0);
  }

  function casino(l: LandmarkDef, facing: Facing): void {
    const { w, d, h } = l.footprint;
    const x = l.position.x;
    const z = l.position.z;
    // Podium with the gaming floor, tower with hotel rooms, palms out front.
    solidBox(ctx, 'deco', x, LOT_Y, z, w, 2.5 * FLOOR, d, 0xfff1c9, rng);
    solidBox(ctx, 'flat', x, LOT_Y + 2.5 * FLOOR, z, w + 0.6, 0.8, d + 0.6, 0xffd23f, rng, { collide: false, topColor: 0xffd23f, sidesOnly: true });
    solidBox(ctx, 'glass', x, LOT_Y + 2.5 * FLOOR, z, w * 0.55, h - 2.5 * FLOOR, d * 0.55, 0xf7a8c4, rng, { glass: true, topColor: 0x3a2f38 });
    solidBox(ctx, 'flat', x, LOT_Y + h, z, w * 0.2, 6, d * 0.2, 0xfff1c9, rng, { collide: false, topColor: 0xfff1c9 });
    const faceLen = facing === 'N' || facing === 'S' ? w : d;
    for (const c of [0xff4fa3, 0x35e6ff, 0xffd23f]) {
      const y = LOT_Y + 2.5 * FLOOR + 0.5 + (c === 0xff4fa3 ? 0 : c === 0x35e6ff ? 0.35 : 0.7);
      faceBox(ctx, 'neon', x, z, w / 2, d / 2, facing, y, faceLen * 0.95, 0.14, 0.14, c, 0, 0.35 + (y - LOT_Y - 2.5 * FLOOR) * 0.3);
    }
    // Vertical neon strips on the tower corners.
    const tw = (w * 0.55) / 2;
    const td = (d * 0.55) / 2;
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) solidBox(ctx, 'neon', x + sx * tw, LOT_Y + 2.5 * FLOOR + 1, z + sz * td, 0.25, h - 2.5 * FLOOR - 2, 0.25, sx * sz > 0 ? 0xff4fa3 : 0x35e6ff, rng, { collide: false, topColor: 0xffffff });
    // Porte-cochère and palms.
    faceBox(ctx, 'flat', x, z, w / 2, d / 2, facing, LOT_Y + 5.5, Math.min(faceLen * 0.5, 22), 0.5, 7, 0xfff1c9, 0, 3.5);
    const n = FACING_NORMAL[facing];
    for (const s of [-1, 1]) {
      const px = x + n.x * (w / 2 + 5) + -n.z * s * 9;
      const pz = z + n.z * (d / 2 + 5) + n.x * s * 9;
      props.add('palm', px, LOT_Y, pz, rng.range(0, 6.28), 1.1);
    }
  }

  function garage(l: LandmarkDef, facing: Facing, trim: number): void {
    const { w, d, h } = l.footprint;
    solidBox(ctx, 'walls', l.position.x, LOT_Y, l.position.z, w, h, d, 0x9a9a98, rng, { uvScale: 25.6 });
    solidBox(ctx, 'flat', l.position.x, LOT_Y + h, l.position.z, w + 0.3, 0.5, d + 0.3, trim, rng, { collide: false, topColor: trim, sidesOnly: true });
    const faceLen = facing === 'N' || facing === 'S' ? w : d;
    // Two roller doors: the drive-in marker sits in front of them.
    for (const s of [-1, 1]) faceBox(ctx, 'flat', l.position.x, l.position.z, w / 2, d / 2, facing, LOT_Y + 2.2, Math.min(6, faceLen * 0.3), 4.4, 0.06, 0x3a3f44, (s * faceLen) / 4, 0);
    faceBox(ctx, 'neon', l.position.x, l.position.z, w / 2, d / 2, facing, LOT_Y + h - 0.5, faceLen * 0.9, 0.12, 0.12, trim, 0, 0.05);
  }

  function gasStation(l: LandmarkDef, facing: Facing): void {
    const { w, d, h } = l.footprint;
    solidBox(ctx, 'flat', l.position.x, LOT_Y, l.position.z, w, h, d, 0xf0f0ec, rng);
    solidBox(ctx, 'flat', l.position.x, LOT_Y + h, l.position.z, w + 0.4, 0.6, d + 0.4, 0xff4d4d, rng, { collide: false, topColor: 0xff4d4d, sidesOnly: true });
    // Canopy west of the store with four pumps and two columns.
    const n = FACING_NORMAL[facing];
    const tx = -n.z;
    const tz = n.x;
    const cx = l.position.x + tx * -(w / 2 + 12) + n.x * 2;
    const cz = l.position.z + tz * -(w / 2 + 12) + n.z * 2;
    solidBox(ctx, 'flat', cx, LOT_Y + 5.2, cz, 18, 0.6, 12, 0xff4d4d, rng, { collide: false, topColor: 0xffffff });
    for (const s of [-1, 1]) solidBox(ctx, 'flat', cx + s * 6, LOT_Y, cz, 0.6, 5.2, 0.6, 0xdddddd, rng);
    for (const s of [-1, 1]) for (const t of [-1, 1]) props.add('pump', cx + s * 3.5, LOT_Y, cz + t * 2.5, 0);
    batches.layer(cx, cz, 'concrete').addQuad(cx - 12, cz - 9, cx + 12, cz + 9, LOT_Y + 0.015, 0x8a8a88, 8);
  }

  function prison(l: LandmarkDef): void {
    const { w, d } = l.footprint;
    const x = l.position.x;
    const z = l.position.z;
    const wallH = 6;
    const t = 0.8;
    // Perimeter wall with a gate gap on the south side, watchtowers at the corners.
    const gate = 8;
    solidBox(ctx, 'flat', x, LOT_Y, z - d / 2, w, wallH, t, 0x9a9a92, rng, { topColor: 0x6a6a64 });
    solidBox(ctx, 'flat', x - (w / 2 + gate / 2) / 2 - gate / 4, LOT_Y, z + d / 2, w / 2 - gate / 2, wallH, t, 0x9a9a92, rng, { topColor: 0x6a6a64 });
    solidBox(ctx, 'flat', x + (w / 2 + gate / 2) / 2 + gate / 4, LOT_Y, z + d / 2, w / 2 - gate / 2, wallH, t, 0x9a9a92, rng, { topColor: 0x6a6a64 });
    solidBox(ctx, 'flat', x - w / 2, LOT_Y, z, t, wallH, d, 0x9a9a92, rng, { topColor: 0x6a6a64 });
    solidBox(ctx, 'flat', x + w / 2, LOT_Y, z, t, wallH, d, 0x9a9a92, rng, { topColor: 0x6a6a64 });
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        const tx = x + (sx * w) / 2;
        const tz = z + (sz * d) / 2;
        solidBox(ctx, 'flat', tx, LOT_Y, tz, 3, 11, 3, 0x7f7f78, rng, { topColor: 0x555550 });
        solidBox(ctx, 'flat', tx, LOT_Y + 11, tz, 4.2, 2.6, 4.2, 0x5f6a72, rng, { collide: false, topColor: 0x3a3f44 });
      }
    }
    // Cell blocks and the administration building behind the gate.
    solidBox(ctx, 'walls', x - 60, LOT_Y, z - 6, 90, 12, 24, 0xb8b4aa, rng, { uvScale: 25.6 });
    solidBox(ctx, 'walls', x + 60, LOT_Y, z - 6, 90, 12, 24, 0xb8b4aa, rng, { uvScale: 25.6 });
    solidBox(ctx, 'walls', x, LOT_Y, z + 12, 30, 8, 18, 0xa8a49a, rng, { uvScale: 12.8 });
    // Gate posts, yard slab and floodlights.
    for (const s of [-1, 1]) solidBox(ctx, 'flat', x + (s * gate) / 2, LOT_Y, z + d / 2, 1.2, 7.5, 1.2, 0x6a6a64, rng);
    batches.layer(x, z, 'concrete').addQuad(x - w / 2 + 1, z - d / 2 + 1, x + w / 2 - 1, z + d / 2 - 1, LOT_Y + 0.01, 0x8d8d8a, 8);
    for (const s of [-1, 1]) props.addLamp(x + s * 12, LOT_Y, z + d / 2 + 4, s > 0 ? Math.PI : 0);
    signBatchAdd(l, new Vector3(x, LOT_Y + 8.4, z + d / 2 + 0.5), 'S', 14, 3.5);
  }

  function addSign(l: LandmarkDef, facing: Facing): void {
    const n = FACING_NORMAL[facing];
    const { w, d, h } = l.footprint;
    const signW = Math.min(14, (facing === 'N' || facing === 'S' ? w : d) * 0.8);
    const y = Math.min(LOT_Y + h - 1.6, LOT_Y + FLOOR * 2 + 1.2);
    const pos = new Vector3(l.position.x + n.x * (w / 2 + 0.2), y, l.position.z + n.z * (d / 2 + 0.2));
    signBatchAdd(l, pos, facing, signW, signW / 4);
  }

  function signBatchAdd(l: LandmarkDef, pos: Vector3, facing: Facing, sw: number, sh: number): void {
    const rect = signs.rects.get(l.id);
    if (!rect) return;
    const n = FACING_NORMAL[facing];
    const tx = -n.z;
    const tz = n.x;
    const hw = sw / 2;
    const hh = sh / 2;
    const c0 = new Vector3(pos.x - tx * hw, pos.y - hh, pos.z - tz * hw);
    const c1 = new Vector3(pos.x + tx * hw, pos.y - hh, pos.z + tz * hw);
    const c2 = new Vector3(pos.x + tx * hw, pos.y + hh, pos.z + tz * hw);
    const c3 = new Vector3(pos.x - tx * hw, pos.y + hh, pos.z - tz * hw);
    const [u0, v0, u1, v1] = rect;
    // With tangent t = (-n.z, n.x) the natural winding faces inward, so reverse it to face `n`.
    signBatch.addQuadCorners([c1, c0, c3, c2], 0xffffff, [
      [u1, v0],
      [u0, v0],
      [u0, v1],
      [u1, v1],
    ]);
    // Backing board so the sign reads from the side too.
    ctx.batches.layer(pos.x, pos.z, 'flat').addBox(pos.x - n.x * 0.12, pos.y, pos.z - n.z * 0.12, n.x === 0 ? sw + 0.3 : 0.2, sh + 0.3, n.x === 0 ? 0.2 : sw + 0.3, 0x15151a);
  }

  function addDoor(l: LandmarkDef, facing: Facing): void {
    // Dark doorway recess with a pale frame at the door marker.
    const n = FACING_NORMAL[facing];
    const { w, d } = l.footprint;
    const tx = -n.z;
    const tz = n.x;
    const along = (l.door.x - l.position.x) * tx + (l.door.z - l.position.z) * tz;
    faceBox(ctx, 'flat', l.position.x, l.position.z, w / 2, d / 2, facing, LOT_Y + 1.35, 2.6, 2.7, 0.08, 0xe8e0d0, along, 0.02);
    faceBox(ctx, 'flat', l.position.x, l.position.z, w / 2, d / 2, facing, LOT_Y + 1.3, 2.2, 2.5, 0.06, 0x1a1c22, along, 0.1);
  }
}
