import { Group, Mesh, Quaternion, Vector3 } from 'three';
import { Config } from '../../core/Config';
import { GeometryBatch } from '../../gfx/GeometryBatch';
import type { CityMaterials } from '../../gfx/Materials';
import type { Physics } from '../../physics/Physics';
import type { Props } from '../Props';
import type { RoadEdge, RoadGraph, RoadNode } from '../RoadGraph';
import type { RoadKind } from '../roadNetwork';
import type { ChunkBatches } from './ChunkBatches';
import type { StaticEntities } from './StaticEntities';

const W = Config.world;
const SW = W.sidewalkWidth;
const CURB_H = W.sidewalkHeight;
const SIDEWALK_COLOR = 0xbdb9b0;
const SIDEWALK_TOP = 0xcfcbc2;
const BRIDGE_COLOR = 0x8f9096;
const RAIL_COLOR = 0xb7bcc4;
const LAMP_SPACING = 32;

export interface RoadBuildOptions {
  graph: RoadGraph;
  batches: ChunkBatches;
  materials: CityMaterials;
  physics: Physics;
  entities: StaticEntities;
  parent: Group;
  props: Props;
}

/**
 * Road surfaces (one merged mesh per road kind), intersection patches, sidewalk
 * strips with curbs, corner pads, crosswalk stripes, bridge decks with railings
 * and piers, plus lamp posts and traffic lights along the network.
 */
export function buildRoads(opts: RoadBuildOptions): { meshes: Mesh[]; triangles: number } {
  const { graph, batches, materials, physics, entities, parent, props } = opts;
  const roadBatches: Record<RoadKind | 'asphalt', GeometryBatch> = {
    street: new GeometryBatch(),
    avenue: new GeometryBatch(),
    highway: new GeometryBatch(),
    dirt: new GeometryBatch(),
    bridge: new GeometryBatch(),
    asphalt: new GeometryBatch(),
  };
  const tmp = { a: new Vector3(), b: new Vector3(), c: new Vector3(), d: new Vector3() };

  for (const edge of graph.edges) buildEdge(edge);
  for (const node of graph.nodes) buildNode(node);

  const meshes: Mesh[] = [];
  let triangles = 0;
  for (const kind of Object.keys(roadBatches) as (RoadKind | 'asphalt')[]) {
    const batch = roadBatches[kind];
    if (batch.isEmpty) continue;
    const geom = batch.build();
    const mesh = new Mesh(geom, materials.roads[kind]);
    mesh.name = `roads:${kind}`;
    mesh.receiveShadow = true;
    mesh.matrixAutoUpdate = false;
    parent.add(mesh);
    meshes.push(mesh);
    triangles += (geom.getIndex()?.count ?? 0) / 3;
  }
  return { meshes, triangles };

  function nodePos(id: number): Vector3 {
    return graph.nodes[id].position;
  }

  function along(edge: RoadEdge, dist: number, lateral: number, yOffset: number, out: Vector3): Vector3 {
    const a = nodePos(edge.a);
    const b = nodePos(edge.b);
    const t = dist / edge.length;
    out.set(a.x + edge.direction.x * dist + edge.right.x * lateral, a.y + (b.y - a.y) * t + yOffset, a.z + edge.direction.z * dist + edge.right.z * lateral);
    return out;
  }

  function buildEdge(edge: RoadEdge): void {
    const half = edge.width / 2;
    const start = edge.trimA;
    const end = edge.length - edge.trimB;
    if (end - start < 0.5) return;
    const batch = roadBatches[edge.kind];
    const c0 = along(edge, start, -half, W.roadSurfaceY, new Vector3());
    const c1 = along(edge, start, half, W.roadSurfaceY, new Vector3());
    const c2 = along(edge, end, half, W.roadSurfaceY, new Vector3());
    const c3 = along(edge, end, -half, W.roadSurfaceY, new Vector3());
    const v0 = start / 12;
    const v1 = end / 12;
    // Counter-clockwise seen from above so the normal points up.
    batch.addQuadCorners([c0, c1, c2, c3], 0xffffff, [
      [0, v0],
      [1, v0],
      [1, v1],
      [0, v1],
    ]);

    if (edge.kind === 'bridge') buildBridge(edge, start, end);
    if (edge.kind === 'street' || edge.kind === 'avenue' || edge.kind === 'bridge') buildSidewalks(edge, start, end, half);
    buildLamps(edge, start, end, half);
  }

  function buildSidewalks(edge: RoadEdge, start: number, end: number, half: number): void {
    const sStart = start + SW;
    const sEnd = end - SW;
    if (sEnd - sStart < 1) return;
    const mid = (sStart + sEnd) / 2;
    const len = sEnd - sStart;
    const rotY = -Math.atan2(edge.direction.z, edge.direction.x);
    const isBridge = edge.kind === 'bridge';
    for (const side of [1, -1] as const) {
      const lateral = side * (half + SW / 2);
      const center = along(edge, mid, lateral, 0, tmp.a);
      const a = nodePos(edge.a);
      const b = nodePos(edge.b);
      const slope = (b.y - a.y) / edge.length;
      const batch = batches.layer(center.x, center.z, 'concrete');
      if (Math.abs(slope) < 1e-4) {
        batch.addBox(center.x, CURB_H / 2, center.z, len, CURB_H, SW, SIDEWALK_COLOR, { rotationY: rotY, uvScale: 8, topColor: SIDEWALK_TOP });
        physics.addStaticCuboid({ x: center.x, y: CURB_H / 2, z: center.z }, { x: len / 2, y: CURB_H / 2, z: SW / 2 }, { entity: entities.sidewalk, rotationY: rotY });
      } else {
        // Sloped walkway on bridge ramps: a rotated box following the incline.
        addInclinedBox(center.x, center.y + CURB_H / 2, center.z, len, CURB_H, SW, rotY, Math.atan(slope), isBridge ? RAIL_COLOR : SIDEWALK_COLOR, batch, entities.bridge);
      }
    }
  }

  function addInclinedBox(cx: number, cy: number, cz: number, len: number, h: number, w: number, rotY: number, pitch: number, color: number, batch: GeometryBatch, entity: StaticEntities['bridge']): void {
    // Build the box axis-aligned along x, then rotate: pitch about z (incline), then yaw.
    const q = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), rotY).multiply(new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), pitch));
    const hx = len / 2;
    const hy = h / 2;
    const hz = w / 2;
    const corners = (y: number): Vector3[] =>
      [
        [-hx, y, -hz],
        [hx, y, -hz],
        [hx, y, hz],
        [-hx, y, hz],
      ].map(([x, yy, z]) => new Vector3(x, yy, z).applyQuaternion(q).add(new Vector3(cx, cy, cz)));
    const top = corners(hy);
    const bottom = corners(-hy);
    batch.addQuadCorners([top[3], top[2], top[1], top[0]], color);
    batch.addQuadCorners([bottom[0], bottom[1], bottom[2], bottom[3]], color);
    batch.addQuadCorners([top[0], top[1], bottom[1], bottom[0]], color);
    batch.addQuadCorners([top[2], top[3], bottom[3], bottom[2]], color);
    batch.addQuadCorners([top[1], top[2], bottom[2], bottom[1]], color);
    batch.addQuadCorners([top[3], top[0], bottom[0], bottom[3]], color);
    physics.addStaticCuboid({ x: cx, y: cy, z: cz }, { x: hx, y: hy, z: hz }, { entity, rotation: { x: q.x, y: q.y, z: q.z, w: q.w } });
  }

  function buildBridge(edge: RoadEdge, start: number, end: number): void {
    const a = nodePos(edge.a);
    const b = nodePos(edge.b);
    const mid = (start + end) / 2;
    const len = end - start;
    const center = along(edge, mid, 0, 0, new Vector3());
    const rotY = -Math.atan2(edge.direction.z, edge.direction.x);
    const pitch = Math.atan((b.y - a.y) / edge.length);
    const deckW = edge.width + 2 * SW;
    const batch = batches.layer(center.x, center.z, 'flat');
    // Deck slab (top face 2 cm under the road surface), guard rails on both edges.
    addInclinedBox(center.x, center.y - 0.6, center.z, len, 1.2, deckW, rotY, pitch, BRIDGE_COLOR, batch, entities.bridge);
    for (const side of [1, -1] as const) {
      const railCenter = along(edge, mid, side * (deckW / 2 - 0.15), 0.75, new Vector3());
      addInclinedBox(railCenter.x, railCenter.y, railCenter.z, len, 1.1, 0.3, rotY, pitch, RAIL_COLOR, batch, entities.bridge);
    }
    // Piers every 30 m where the deck is above the water.
    for (let d = start + 15; d < end; d += 30) {
      const p = along(edge, d, 0, 0, tmp.b);
      if (p.y < 2) continue;
      const pierBatch = batches.layer(p.x, p.z, 'flat');
      const h = p.y - 0.6 - W.seabedDepth;
      pierBatch.addBox(p.x, W.seabedDepth + h / 2, p.z, 3, h, deckW - 4, 0x6f7176);
      physics.addStaticCuboid({ x: p.x, y: W.seabedDepth + h / 2, z: p.z }, { x: 1.5, y: h / 2, z: (deckW - 4) / 2 }, { entity: entities.bridge });
    }
  }

  function buildLamps(edge: RoadEdge, start: number, end: number, half: number): void {
    if (edge.kind === 'dirt') return;
    const usable = end - start - 2 * SW;
    if (usable < LAMP_SPACING * 0.6) return;
    const count = Math.max(1, Math.round(usable / LAMP_SPACING));
    const step = usable / count;
    const isHighway = edge.kind === 'highway';
    for (let i = 0; i < count; i++) {
      const d = start + SW + step * (i + 0.5);
      const side = i % 2 === 0 ? 1 : -1;
      const lateral = side * (isHighway ? half + 1.2 : half + SW - 0.6);
      const p = along(edge, d, lateral, 0, tmp.c);
      const ground = edge.kind === 'bridge' ? p.y + CURB_H : CURB_H;
      // The arm points back toward the road center.
      props.addLamp(p.x, ground, p.z, yawFor(-edge.right.x * side, -edge.right.z * side));
      if (isHighway) {
        const q = along(edge, d, -lateral, 0, tmp.d);
        props.addLamp(q.x, ground, q.z, yawFor(edge.right.x * side, edge.right.z * side));
      }
    }
  }

  function buildNode(node: RoadNode): void {
    if (node.halfNS > 0 && node.halfEW > 0) {
      const p = node.position;
      roadBatches.asphalt.addQuad(p.x - node.halfNS, p.z - node.halfEW, p.x + node.halfNS, p.z + node.halfEW, p.y + W.roadSurfaceY, 0xffffff, 12);
    }
    const edges = node.edges.map((id) => graph.edges[id]);
    const walkable = edges.filter((e) => e.kind === 'street' || e.kind === 'avenue' || e.kind === 'bridge');
    if (walkable.length === 0) return;
    buildCornerPads(node, edges);
    buildCrosswalks(node, walkable);
    if (node.hasLight) buildSignals(node, edges);
  }

  /** Direction from the node into an edge. */
  function outDir(node: RoadNode, edge: RoadEdge, out: Vector3): Vector3 {
    return edge.a === node.id ? out.copy(edge.direction) : out.copy(edge.direction).negate();
  }

  function buildCornerPads(node: RoadNode, edges: RoadEdge[]): void {
    const p = node.position;
    const hasDir = { e: false, w: false, n: false, s: false };
    let kindE: RoadKind | null = null;
    let kindW: RoadKind | null = null;
    let kindN: RoadKind | null = null;
    let kindS: RoadKind | null = null;
    for (const e of edges) {
      const d = outDir(node, e, tmp.a);
      if (d.x > 0.5) (hasDir.e = true), (kindE = e.kind);
      else if (d.x < -0.5) (hasDir.w = true), (kindW = e.kind);
      else if (d.z > 0.5) (hasDir.s = true), (kindS = e.kind);
      else if (d.z < -0.5) (hasDir.n = true), (kindN = e.kind);
    }
    const hx = node.halfNS;
    const hz = node.halfEW;
    const batch = batches.layer(p.x, p.z, 'concrete');
    const pad = (cx: number, cz: number, sx: number, sz: number): void => {
      batch.addBox(cx, p.y + CURB_H / 2, cz, sx, CURB_H, sz, SIDEWALK_COLOR, { uvScale: 8, topColor: SIDEWALK_TOP });
      physics.addStaticCuboid({ x: cx, y: p.y + CURB_H / 2, z: cz }, { x: sx / 2, y: CURB_H / 2, z: sz / 2 }, { entity: entities.sidewalk });
    };
    const walk = (k: RoadKind | null): boolean => k !== null && k !== 'highway' && k !== 'dirt';
    // Four corner pads.
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        const ew = sx > 0 ? kindE : kindW;
        const ns = sz > 0 ? kindS : kindN;
        if (!walk(ew) && !walk(ns) && !(hx > 0 && hz > 0)) continue;
        pad(p.x + sx * (hx + SW / 2), p.z + sz * (hz + SW / 2), SW, SW);
      }
    }
    // Closing strips on sides without a road (dead ends, T-junctions).
    if (!hasDir.n && hz > 0 && (walk(kindE) || walk(kindW))) pad(p.x, p.z - hz - SW / 2, hx * 2, SW);
    if (!hasDir.s && hz > 0 && (walk(kindE) || walk(kindW))) pad(p.x, p.z + hz + SW / 2, hx * 2, SW);
    if (!hasDir.e && hx > 0 && (walk(kindN) || walk(kindS))) pad(p.x + hx + SW / 2, p.z, SW, hz * 2);
    if (!hasDir.w && hx > 0 && (walk(kindN) || walk(kindS))) pad(p.x - hx - SW / 2, p.z, SW, hz * 2);
  }

  function buildCrosswalks(node: RoadNode, edges: RoadEdge[]): void {
    if (edges.length < 3 && node.edges.length < 3) return;
    const p = node.position;
    const batch = batches.layer(p.x, p.z, 'flat');
    for (const e of edges) {
      const dir = outDir(node, e, tmp.a);
      const trim = e.a === node.id ? e.trimA : e.trimB;
      const half = e.width / 2;
      const rotY = -Math.atan2(dir.z, dir.x);
      const stripes = Math.floor((half * 2 - 1) / 1.0);
      for (let i = 0; i < stripes; i++) {
        const lateral = -half + 0.8 + i * 1.0;
        const cx = p.x + dir.x * (trim + SW / 2) + e.right.x * lateral;
        const cz = p.z + dir.z * (trim + SW / 2) + e.right.z * lateral;
        batch.addBox(cx, p.y + W.roadSurfaceY + 0.012, cz, 2.6, 0.01, 0.5, 0xd9d9d0, { rotationY: rotY });
      }
    }
  }

  function buildSignals(node: RoadNode, edges: RoadEdge[]): void {
    const p = node.position;
    for (const e of edges) {
      if (e.kind === 'dirt') continue;
      const dir = outDir(node, e, tmp.a);
      const trim = e.a === node.id ? e.trimA : e.trimB;
      const half = e.width / 2;
      // Pole on the near-right corner pad of the approach (traffic approaching the node drives on the
      // right side of its lane, i.e. on the -right side of `dir` when dir points away from the node).
      const lateral = -(half + SW / 2);
      const x = p.x + dir.x * (trim + SW / 2) + e.right.x * lateral;
      const z = p.z + dir.z * (trim + SW / 2) + e.right.z * lateral;
      const axis: 'ns' | 'ew' = Math.abs(dir.z) > Math.abs(dir.x) ? 'ns' : 'ew';
      // Arm extends over the road, perpendicular to the approach, toward the road center.
      const rotY = yawFor(-e.right.x * lateral, -e.right.z * lateral);
      const green = (node.lightPhase === 0) === (axis === 'ns');
      props.addTrafficLight(node.id, axis, x, p.y + CURB_H, z, rotY, green);
    }
  }
}

/** Yaw (rotation about Y) that maps the local +X axis onto the horizontal direction (dx, dz). */
export function yawFor(dx: number, dz: number): number {
  return Math.atan2(-dz, dx);
}
