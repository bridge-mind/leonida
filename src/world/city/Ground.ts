import { BufferAttribute, Color, Group, Mesh, PlaneGeometry } from 'three';
import { Config } from '../../core/Config';
import type { Rect } from '../../data/districts';
import type { CityMaterials } from '../../gfx/Materials';
import type { Physics } from '../../physics/Physics';
import { Groups } from '../../physics/Physics';
import type { Districts } from '../Districts';
import type { StaticEntities } from './StaticEntities';

const W = Config.world;

export interface GroundOptions {
  physics: Physics;
  materials: CityMaterials;
  districts: Districts;
  entities: StaticEntities;
  parent: Group;
}

/** Land rectangles in world space; everything outside them is water. */
export const LAND_RECTS: { rect: Rect; kind: 'mainland' | 'island' | 'swamp' }[] = [
  { rect: W.mainland, kind: 'mainland' },
  { rect: W.island, kind: 'island' },
  { rect: W.swamp, kind: 'swamp' },
];

export function isOnLand(x: number, z: number): boolean {
  for (const l of LAND_RECTS) {
    const r = l.rect;
    if (x >= r.x0 && x <= r.x1 && z >= r.z0 && z <= r.z1) return true;
  }
  return false;
}

/**
 * Ground meshes (vertex-colored by district) and colliders: one box per land rect
 * plus the seabed under everything.
 */
export function buildGround(opts: GroundOptions): { meshes: Mesh[] } {
  const { physics, materials, districts, entities, parent } = opts;
  const meshes: Mesh[] = [];
  const color = new Color();
  const tmp = new Color();

  for (const land of LAND_RECTS) {
    const r = land.rect;
    const w = r.x1 - r.x0;
    const d = r.z1 - r.z0;
    const cell = 25;
    const geom = new PlaneGeometry(w, d, Math.ceil(w / cell), Math.ceil(d / cell));
    geom.rotateX(-Math.PI / 2);
    const pos = geom.getAttribute('position');
    const colors = new Float32Array(pos.count * 3);
    const cx = (r.x0 + r.x1) / 2;
    const cz = (r.z0 + r.z1) / 2;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i) + cx;
      const z = pos.getZ(i) + cz;
      const district = districts.at(x, z) ?? districts.nearest(x, z);
      color.set(district.groundColor);
      // Highway verges and inter-district gaps read as scrub grass.
      if (!districts.at(x, z)) color.lerp(tmp.set(0x6e8a4a), 0.6);
      const n = 0.92 + 0.08 * hash(x, z);
      colors[i * 3] = color.r * n;
      colors[i * 3 + 1] = color.g * n;
      colors[i * 3 + 2] = color.b * n;
    }
    geom.setAttribute('color', new BufferAttribute(colors, 3));
    // Scale UVs so the 4 m noise texture tiles at a pleasant density.
    const uv = geom.getAttribute('uv');
    for (let i = 0; i < uv.count; i++) uv.setXY(i, (uv.getX(i) * w) / 8, (uv.getY(i) * d) / 8);
    const material = land.kind === 'swamp' ? materials.grass : land.kind === 'island' ? materials.concrete : materials.grass;
    const mesh = new Mesh(geom, material);
    mesh.position.set(cx, 0, cz);
    mesh.receiveShadow = true;
    mesh.name = `ground:${land.kind}`;
    mesh.matrixAutoUpdate = false;
    mesh.updateMatrix();
    parent.add(mesh);
    meshes.push(mesh);

    physics.addStaticCuboid(
      { x: cx, y: -1, z: cz },
      { x: w / 2, y: 1, z: d / 2 },
      { entity: land.kind === 'swamp' ? entities.dirt : entities.ground, groups: Groups.STATIC },
    );
  }

  // Seabed keeps anything that falls into water from falling forever.
  const half = W.size * 1.2;
  physics.addStaticCuboid({ x: 0, y: W.seabedDepth - 1, z: 0 }, { x: half, y: 1, z: half }, { entity: entities.seabed, groups: Groups.STATIC });
  return { meshes };
}

function hash(x: number, z: number): number {
  const s = Math.sin(x * 12.9898 + z * 78.233) * 43758.5453;
  return s - Math.floor(s);
}
