import { BufferAttribute, Group, Mesh, PlaneGeometry } from 'three';
import { Config } from '../../core/Config';
import type { Random } from '../../core/Random';
import { DISTRICT_BY_ID } from '../../data/districts';
import { addBuilding, solidBox, type BuildContext, type Lot } from '../../gfx/BuildingBuilder';
import type { CityMaterials } from '../../gfx/Materials';
import type { Physics } from '../../physics/Physics';
import type { Props } from '../Props';
import type { RoadGraph } from '../RoadGraph';
import { SWAMP_POOLS } from '../Water';
import type { ChunkBatches } from './ChunkBatches';
import { isOnLand } from './Ground';
import type { StaticEntities } from './StaticEntities';

const W = Config.world;
const LOT_Y = W.lotSurfaceY;

export interface SpecialAreaOptions {
  batches: ChunkBatches;
  physics: Physics;
  entities: StaticEntities;
  props: Props;
  rng: Random;
  materials: CityMaterials;
  parent: Group;
  graph: RoadGraph;
}

/**
 * Hand-authored set pieces: Vice Beach sand/boardwalk/pier and bayfront hotels,
 * the Docks cranes and marina, Grassrivers shacks and reeds, the Port Gellhorn
 * trailer park, plus street furniture along every sidewalk.
 */
export function buildSpecialAreas(opts: SpecialAreaOptions): { meshes: Mesh[]; buildings: number } {
  const { batches, physics, entities, props, rng, materials, parent, graph } = opts;
  const ctx: BuildContext = { batches, physics, entities };
  const meshes: Mesh[] = [];
  let buildings = 0;

  beach();
  bayfront();
  docks();
  swamp();
  trailerPark();
  streetFurniture();
  return { meshes, buildings };

  function beach(): void {
    // Sand strip east of the boardwalk down to the water line.
    const sand = new Mesh(new PlaneGeometry(W.island.x1 - 878, W.island.z1 - W.island.z0, 1, 1), materials.sand);
    sand.rotation.x = -Math.PI / 2;
    sand.position.set((878 + W.island.x1) / 2, 0.03, 0);
    sand.receiveShadow = true;
    sand.name = 'beachSand';
    const uv = sand.geometry.getAttribute('uv');
    for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * 8, uv.getY(i) * 160);
    sand.geometry.setAttribute('color', new BufferAttribute(new Float32Array(uv.count * 3).fill(1), 3));
    sand.matrixAutoUpdate = false;
    sand.updateMatrix();
    parent.add(sand);
    meshes.push(sand);

    // Boardwalk: raised planks between Ocean Drive's sidewalk and the sand.
    for (let z = W.island.z0 + 10; z < W.island.z1 - 10; z += 40) {
      const len = Math.min(40, W.island.z1 - 10 - z);
      batches.layer(874, z + len / 2, 'planks').addBox(874, 0.2, z + len / 2, 8, 0.4, len, 0xa57a4e, { uvScale: 4, topColor: 0xb8895a });
      physics.addStaticCuboid({ x: 874, y: 0.2, z: z + len / 2 }, { x: 4, y: 0.2, z: len / 2 }, { entity: entities.wood });
    }
    for (let z = W.island.z0 + 20; z < W.island.z1 - 10; z += 14) {
      props.add('palm', 871 + rng.range(-0.5, 0.5), 0.4, z + rng.range(-3, 3), rng.range(0, 6.28), rng.range(0.85, 1.15));
      if (rng.chance(0.35)) props.add('bench', 877, 0.4, z + 5, Math.PI / 2);
    }
    for (let z = W.island.z0 + 70; z < W.island.z1 - 40; z += 150) props.add('lifeguard', 915, 0.03, z, Math.PI / 2);
    const umbrellaColors = [0xff4fa3, 0x35e6ff, 0xffd23f, 0xffffff, 0xff7a3a];
    for (let i = 0; i < 90; i++) {
      const z = rng.range(W.island.z0 + 20, W.island.z1 - 20);
      props.add('umbrella', rng.range(884, 926), 0.03, z, 0, rng.range(0.8, 1.1), rng.pick(umbrellaColors));
    }

    // Pier at z = 0 running east into the ocean on piles, with railings and a pavilion.
    const pierX0 = 878;
    const pierX1 = 985;
    const pierY = 1.4;
    const pierBatch = batches.layer(930, 0, 'planks');
    pierBatch.addBox((pierX0 + pierX1) / 2, pierY, 0, pierX1 - pierX0, 0.3, 9, 0xa57a4e, { uvScale: 4, topColor: 0xb8895a });
    physics.addStaticCuboid({ x: (pierX0 + pierX1) / 2, y: pierY, z: 0 }, { x: (pierX1 - pierX0) / 2, y: 0.15, z: 4.5 }, { entity: entities.wood });
    for (const s of [-1, 1]) {
      pierBatch.addBox((pierX0 + pierX1) / 2, pierY + 0.7, s * 4.3, pierX1 - pierX0, 1.0, 0.15, 0xd8c8a8);
      physics.addStaticCuboid({ x: (pierX0 + pierX1) / 2, y: pierY + 0.7, z: s * 4.3 }, { x: (pierX1 - pierX0) / 2, y: 0.5, z: 0.08 }, { entity: entities.wood });
    }
    for (let x = pierX0 + 6; x < pierX1; x += 8) {
      for (const s of [-1, 1]) {
        const h = pierY - W.seabedDepth;
        batches.layer(x, 0, 'flat').addBox(x, W.seabedDepth + h / 2, s * 3.5, 0.5, h, 0.5, 0x5a4a3a);
      }
    }
    solidBox(ctx, 'flat', pierX1 - 8, pierY + 0.15, 0, 10, 4, 8, 0x8fe3dc, rng, { topColor: 0xf7a8c4 });
    batches.layer(pierX1 - 8, 0, 'flat').addGableRoof(pierX1 - 8, pierY + 4.15, 0, 11, 9, 1.6, 0xf7a8c4, false);
    // Ramp from the boardwalk up onto the pier deck.
    pierBatch.addBox(880, 0.8, 0, 6, 0.2, 9, 0xa57a4e, { topColor: 0xb8895a });
    physics.addStaticCuboid({ x: 880, y: 0.75, z: 0 }, { x: 3.2, y: 0.12, z: 4.5 }, { entity: entities.wood, rotation: pitchQuat(-0.2) });
  }

  function bayfront(): void {
    // Hotels between the bay shore (x 500) and Bay Road (x 560), facing east onto the road.
    const d = DISTRICT_BY_ID.viceBeach;
    for (let z = -600; z < 600; z += 100) {
      const lot: Lot = { x0: 506, x1: 550, z0: z + 10, z1: z + 100 - 10, facing: 'E', district: d };
      // Split the 80 m block face into two hotel lots.
      for (const [a, b] of [
        [lot.z0, lot.z0 + 38],
        [lot.z1 - 38, lot.z1],
      ]) {
        batches.layer(528, (a + b) / 2, 'concrete').addQuad(lot.x0, a, lot.x1, b, LOT_Y, d.lotColor, 8);
        if (addBuilding(ctx, { ...lot, z0: a, z1: b }, 'artDeco', rng)) buildings++;
      }
      props.add('palm', 552, LOT_Y, z + 50, rng.range(0, 6.28));
    }
    // Seawall along the whole west shore of the island.
    batches.layer(502, 0, 'concrete').addBox(502, 0.3, 0, 4, 0.6, W.island.z1 - W.island.z0, 0x9a9a92, { uvScale: 8, topColor: 0xb0b0a8 });
    physics.addStaticCuboid({ x: 502, y: 0.3, z: 0 }, { x: 2, y: 0.3, z: (W.island.z1 - W.island.z0) / 2 }, { entity: entities.sidewalk });
  }

  function docks(): void {
    // Gantry cranes along the southern quay.
    for (const x of [-660, -520, -380]) crane(x, 880);
    // Quay slab and bollards along the south shore of the mainland.
    batches.layer(-400, 890, 'concrete').addQuad(-760, 862, -140, 900, LOT_Y, 0x7a7a76, 8);
    for (let x = -740; x < -160; x += 25) props.add('bollard', x, LOT_Y, 897, 0);
    // Marina: two piers with moored boats south of the marina office.
    for (const px of [-125, -55]) {
      const z0 = 892;
      const z1 = 955;
      const pb = batches.layer(px, (z0 + z1) / 2, 'planks');
      pb.addBox(px, 0.7, (z0 + z1) / 2, 4, 0.3, z1 - z0, 0xa57a4e, { uvScale: 4, topColor: 0xb8895a });
      physics.addStaticCuboid({ x: px, y: 0.7, z: (z0 + z1) / 2 }, { x: 2, y: 0.15, z: (z1 - z0) / 2 }, { entity: entities.wood });
      for (let z = z0 + 8; z < z1 - 6; z += 12) {
        for (const s of [-1, 1]) {
          if (rng.chance(0.3)) continue;
          boat(px + s * 5.5, z, rng);
        }
      }
    }
    for (let x = -140; x < -30; x += 12) props.add('bollard', x, LOT_Y, 889, 0);
  }

  function crane(x: number, z: number): void {
    const legH = 26;
    const span = 34;
    const color = 0xd9a52c;
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) solidBox(ctx, 'flat', x + sx * 6, LOT_Y, z + sz * 9, 1.4, legH, 1.4, color, rng, { topColor: color });
    for (const sz of [-1, 1]) solidBox(ctx, 'flat', x, LOT_Y + legH, z + sz * 9, span, 2.2, 1.6, color, rng, { collide: false, topColor: color });
    solidBox(ctx, 'flat', x + 6, LOT_Y + legH - 3.5, z, 4, 3, 4, 0x3a3f46, rng, { collide: false, topColor: 0x3a3f46 });
    solidBox(ctx, 'flat', x, LOT_Y + legH + 2.2, z, span * 0.9, 0.8, 0.8, 0xbfc4cc, rng, { collide: false, topColor: 0xff3030 });
  }

  function boat(x: number, z: number, r: Random): void {
    const hull = r.pick([0xffffff, 0xf0f0f0, 0x2b4c8f, 0xd94a4a]);
    const batch = batches.layer(x, z, 'flat');
    batch.addBox(x, 0.1, z, 3.2, 1.0, 8.5, hull, { topColor: 0xe8e0d0 });
    batch.addBox(x, 0.9, z - 1, 2.4, 1.2, 3.0, 0xf5f5f5, { topColor: 0xd8d8d8 });
    batch.addBox(x, 1.6, z - 1, 2.2, 0.6, 2.2, 0x35e6ff, { topColor: 0xffffff });
    physics.addStaticCuboid({ x, y: 0.4, z }, { x: 1.6, y: 0.6, z: 4.25 }, { entity: entities.wood });
  }

  function swamp(): void {
    const d = DISTRICT_BY_ID.grassrivers;
    // Reeds and cypress around every pool, shacks along the dirt roads.
    for (const pool of SWAMP_POOLS) {
      const perimeter = 2 * (pool.x1 - pool.x0 + pool.z1 - pool.z0);
      const n = Math.floor(perimeter / 4);
      for (let i = 0; i < n; i++) {
        const t = rng.next();
        const edge = rng.int(0, 3);
        const x = edge < 2 ? pool.x0 + t * (pool.x1 - pool.x0) : edge === 2 ? pool.x0 : pool.x1;
        const z = edge >= 2 ? pool.z0 + t * (pool.z1 - pool.z0) : edge === 0 ? pool.z0 : pool.z1;
        const jitter = rng.range(-1.5, 2.5);
        const px = x + (edge === 2 ? -jitter : edge === 3 ? jitter : rng.range(-1, 1));
        const pz = z + (edge === 0 ? -jitter : edge === 1 ? jitter : rng.range(-1, 1));
        if (!isOnLand(px, pz)) continue;
        props.add('reeds', px, 0.05, pz, rng.range(0, 6.28), rng.range(0.8, 1.3));
      }
      for (let i = 0; i < 4; i++) {
        const px = rng.chance(0.5) ? pool.x0 - rng.range(4, 12) : pool.x1 + rng.range(4, 12);
        const pz = rng.range(pool.z0, pool.z1);
        if (px > -808 || !isOnLand(px, pz) || nearRoad(px, pz, 8)) continue;
        props.add('cypress', px, 0.03, pz, rng.range(0, 6.28), rng.range(0.8, 1.2));
      }
    }
    for (const line of [
      { z: -600, x0: -900, x1: -770 },
      { z: -200, x0: -960, x1: -770 },
      { z: 300, x0: -960, x1: -770 },
    ]) {
      for (let x = line.x0 + 15; x < line.x1 - 10; x += rng.range(28, 44)) {
        const side = rng.sign();
        const z0 = line.z + side * 8;
        const z1 = line.z + side * 20;
        const lot: Lot = { x0: x - 5, x1: x + 5, z0: Math.min(z0, z1), z1: Math.max(z0, z1), facing: side > 0 ? 'N' : 'S', district: d };
        if (inPool(lot.x0, lot.z0) || inPool(lot.x1, lot.z1)) continue;
        if (addBuilding(ctx, lot, 'swamp', rng)) buildings++;
      }
    }
    // Airboat dock at the end of Airboat Road, and scattered trees along the highway verge.
    const dockBatch = batches.layer(-975, -200, 'planks');
    dockBatch.addBox(-975, 0.35, -200, 26, 0.3, 5, 0xa57a4e, { uvScale: 4, topColor: 0xb8895a });
    physics.addStaticCuboid({ x: -975, y: 0.35, z: -200 }, { x: 13, y: 0.15, z: 2.5 }, { entity: entities.wood });
    for (let i = 0; i < 60; i++) {
      const x = rng.range(-995, -810);
      const z = rng.range(-880, 880);
      if (inPool(x, z) || nearRoad(x, z, 7) || !isOnLand(x, z)) continue;
      props.add(rng.chance(0.6) ? 'cypress' : 'reeds', x, 0.03, z, rng.range(0, 6.28), rng.range(0.7, 1.2));
    }
  }

  function inPool(x: number, z: number): boolean {
    return SWAMP_POOLS.some((p) => x >= p.x0 - 2 && x <= p.x1 + 2 && z >= p.z0 - 2 && z <= p.z1 + 2);
  }

  function nearRoad(x: number, z: number, margin: number): boolean {
    const lane = graph.nearestLanePoint({ x, y: 0, z });
    if (!lane) return false;
    return Math.hypot(lane.point.x - x, lane.point.z - z) < lane.edge.width / 2 + margin;
  }

  function trailerPark(): void {
    // Reserved block in Port Gellhorn: rows of trailers on gravel with a few trees.
    const x0 = -590;
    const x1 = -410;
    const z0 = -650;
    const z1 = -540;
    batches.layer(-500, -595, 'concrete').addQuad(x0, z0, x1, z1, LOT_Y + 0.01, 0x9a9484, 8);
    const colors = [0xf4f0e6, 0xd9d3c3, 0xb8c9d9, 0xe0e6dc, 0xd4b896, 0xc9c0a8];
    for (let z = z0 + 12; z < z1 - 8; z += 22) {
      for (let x = x0 + 10; x < x1 - 8; x += 16) {
        if (rng.chance(0.2)) continue;
        const c = rng.pick(colors);
        const rot = rng.range(-0.15, 0.15);
        const batch = batches.layer(x, z, 'flat');
        batch.addBox(x, LOT_Y + 0.4, z, 3.2, 0.8, 10, 0x555555, { rotationY: rot });
        batch.addBox(x, LOT_Y + 2.0, z, 3.4, 2.4, 10.5, c, { rotationY: rot, topColor: 0x8a8a8a });
        batch.addBox(x + 1.75, LOT_Y + 1.2, z + 2, 0.06, 1.8, 0.9, 0x2a2a33, { rotationY: rot });
        physics.addStaticCuboid({ x, y: LOT_Y + 1.6, z }, { x: 1.7, y: 1.6, z: 5.25 }, { entity: entities.metalProp, rotationY: rot });
        if (rng.chance(0.4)) props.add('dumpster', x + 3, LOT_Y, z - 6, rng.range(0, 6.28));
      }
    }
    for (let i = 0; i < 8; i++) props.add('cypress', rng.range(x0 + 4, x1 - 4), LOT_Y, rng.range(z0 + 2, z1 - 2), rng.range(0, 6.28), 0.7);
  }

  function streetFurniture(): void {
    for (const sw of graph.sidewalks) {
      const edge = graph.edges[sw.edgeId];
      const a = sw.points[0];
      const b = sw.points[1];
      const len = a.distanceTo(b);
      if (len < 20) continue;
      const dir = b.clone().sub(a).normalize();
      // Outward normal of the sidewalk (away from the road).
      const out = edge.right.clone().multiplyScalar(sw.side);
      const isBeach = a.x > 500;
      const isDowntownCore = Math.abs(a.x) < 260 && a.z > -400 && a.z < 300;
      const spacing = isBeach ? 16 : 30;
      for (let t = 10; t < len - 8; t += spacing) {
        const px = a.x + dir.x * t + out.x * 1.4;
        const pz = a.z + dir.z * t + out.z * 1.4;
        const yaw = Math.atan2(-out.z, out.x);
        const roll = rng.next();
        if (isBeach && roll < 0.55) props.add('palm', px, W.sidewalkHeight, pz, rng.range(0, 6.28), rng.range(0.8, 1.1));
        else if (roll < 0.25) props.add('bench', px - out.x * 0.6, W.sidewalkHeight, pz - out.z * 0.6, yaw + Math.PI / 2);
        else if (roll < 0.4) props.add('hydrant', px, W.sidewalkHeight, pz, 0);
        else if (roll < 0.5 && !isDowntownCore) props.add('dumpster', px, W.sidewalkHeight, pz, yaw);
        else if (roll < 0.62 && isDowntownCore) props.add('palm', px, W.sidewalkHeight, pz, rng.range(0, 6.28), 0.9);
      }
    }
  }

  function pitchQuat(angle: number): { x: number; y: number; z: number; w: number } {
    return { x: 0, y: 0, z: Math.sin(angle / 2), w: Math.cos(angle / 2) };
  }
}
