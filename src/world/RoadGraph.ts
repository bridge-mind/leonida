import { Vector3 } from 'three';
import type { V3 } from '../core/entities';
import type { Random } from '../core/Random';
import type { RoadKind, RoadSurface } from './roadNetwork';

export interface RoadNode {
  id: number;
  position: Vector3;
  /** Incident edge ids. */
  edges: number[];
  /** 0 or 1: which axis is green first (traffic system cycles from here). */
  lightPhase: number;
  hasLight: boolean;
  /** Half width of the widest N–S / E–W road meeting here (0 if none). */
  halfNS: number;
  halfEW: number;
}

export interface RoadEdge {
  id: number;
  a: number;
  b: number;
  kind: RoadKind;
  lanesPerDirection: number;
  width: number;
  speedLimit: number;
  isHighway: boolean;
  surface: RoadSurface;
  length: number;
  /** Unit direction a→b (horizontal). */
  direction: Vector3;
  /** Unit right-hand perpendicular of `direction`. */
  right: Vector3;
  /** Lane ids traveling a→b then b→a. */
  lanes: number[];
  /** Sidewalk ids (right side of a→b first), empty on highways/dirt. */
  sidewalks: number[];
  /** Distance from node a/b to the start of the drivable segment (intersection half width). */
  trimA: number;
  trimB: number;
}

export interface LanePolyline {
  id: number;
  edgeId: number;
  from: number;
  to: number;
  /** 0 = nearest the center line. */
  laneIndex: number;
  points: Vector3[];
}

export interface SidewalkPolyline {
  id: number;
  edgeId: number;
  side: -1 | 1;
  points: Vector3[];
}

export interface Crosswalk {
  nodeId: number;
  edgeId: number;
  a: Vector3;
  b: Vector3;
}

export interface ParkingSpot {
  position: Vector3;
  heading: number;
  edgeId: number;
  side: -1 | 1;
  kind: 'curb' | 'lot';
}

export interface LanePoint {
  point: Vector3;
  direction: Vector3;
  lane: LanePolyline;
  edge: RoadEdge;
  segment: number;
  t: number;
}

export interface SidewalkPoint {
  point: Vector3;
  direction: Vector3;
  sidewalk: SidewalkPolyline;
  segment: number;
  t: number;
}

interface Segment {
  polyline: number;
  index: number;
  ax: number;
  az: number;
  bx: number;
  bz: number;
  ay: number;
  by: number;
}

/**
 * The spine of the game: nodes, edges, derived lane/sidewalk polylines, crosswalks
 * and parking spots, with nearest-point queries, random points and A*.
 * Build one with `buildRoadGraph()` from `RoadGraphBuilder.ts`.
 */
export class RoadGraph {
  readonly nodes: RoadNode[] = [];
  readonly edges: RoadEdge[] = [];
  readonly lanes: LanePolyline[] = [];
  readonly sidewalks: SidewalkPolyline[] = [];
  readonly crosswalks: Crosswalk[] = [];
  readonly parkingSpots: ParkingSpot[] = [];

  private laneGrid = new SegmentGrid(60);
  private sidewalkGrid = new SegmentGrid(60);
  private laneSegments: Segment[] = [];
  private sidewalkSegments: Segment[] = [];
  private readonly edgeByPair = new Map<number, RoadEdge>();

  /** Called by the builder after nodes/edges/lanes/sidewalks are filled. */
  finalize(): void {
    this.laneSegments = collectSegments(this.lanes.map((l) => l.points));
    this.sidewalkSegments = collectSegments(this.sidewalks.map((s) => s.points));
    this.laneGrid.build(this.laneSegments);
    this.sidewalkGrid.build(this.sidewalkSegments);
    this.edgeByPair.clear();
    for (const e of this.edges) this.edgeByPair.set(pairKey(e.a, e.b), e);
  }

  edgeBetween(a: number, b: number): RoadEdge | undefined {
    return this.edgeByPair.get(pairKey(a, b));
  }

  otherNode(edge: RoadEdge, nodeId: number): number {
    return edge.a === nodeId ? edge.b : edge.a;
  }

  // --- Nearest queries ----------------------------------------------------------

  nearestNode(pos: V3): RoadNode {
    let best = this.nodes[0];
    let bestD = Infinity;
    for (const n of this.nodes) {
      const dx = n.position.x - pos.x;
      const dz = n.position.z - pos.z;
      const d = dx * dx + dz * dz;
      if (d < bestD) {
        bestD = d;
        best = n;
      }
    }
    return best;
  }

  nearestLanePoint(pos: V3): LanePoint | null {
    const hit = this.laneGrid.nearest(pos.x, pos.z, this.laneSegments);
    if (!hit) return null;
    const seg = this.laneSegments[hit.index];
    const lane = this.lanes[seg.polyline];
    return { point: hit.point, direction: hit.direction, lane, edge: this.edges[lane.edgeId], segment: seg.index, t: hit.t };
  }

  nearestSidewalkPoint(pos: V3): SidewalkPoint | null {
    const hit = this.sidewalkGrid.nearest(pos.x, pos.z, this.sidewalkSegments);
    if (!hit) return null;
    const seg = this.sidewalkSegments[hit.index];
    return { point: hit.point, direction: hit.direction, sidewalk: this.sidewalks[seg.polyline], segment: seg.index, t: hit.t };
  }

  /** Random point on a lane whose distance to `pos` is within [minR, maxR]; null if none. */
  randomLanePointNear(pos: V3, minR: number, maxR: number, rng: Random, filter?: (edge: RoadEdge) => boolean): LanePoint | null {
    const seg = this.laneGrid.randomInRing(pos.x, pos.z, minR, maxR, this.laneSegments, rng, (s) =>
      filter ? filter(this.edges[this.lanes[s.polyline].edgeId]) : true,
    );
    if (!seg) return null;
    const lane = this.lanes[seg.segment.polyline];
    return { point: seg.point, direction: seg.direction, lane, edge: this.edges[lane.edgeId], segment: seg.segment.index, t: seg.t };
  }

  randomSidewalkPointNear(pos: V3, minR: number, maxR: number, rng: Random): SidewalkPoint | null {
    const seg = this.sidewalkGrid.randomInRing(pos.x, pos.z, minR, maxR, this.sidewalkSegments, rng, () => true);
    if (!seg) return null;
    return { point: seg.point, direction: seg.direction, sidewalk: this.sidewalks[seg.segment.polyline], segment: seg.segment.index, t: seg.t };
  }

  // --- Pathfinding -------------------------------------------------------------

  /** A* over nodes. Returns node ids from `from` to `to` inclusive, or null if unreachable. */
  astar(from: number, to: number, weight: 'distance' | 'time' = 'distance'): number[] | null {
    if (from === to) return [from];
    const n = this.nodes.length;
    const g = new Float64Array(n).fill(Infinity);
    const parent = new Int32Array(n).fill(-1);
    const closed = new Uint8Array(n);
    const target = this.nodes[to].position;
    const heap = new MinHeap();
    g[from] = 0;
    heap.push(from, this.heuristic(from, target, weight));
    while (heap.size > 0) {
      const current = heap.pop();
      if (current === to) break;
      if (closed[current]) continue;
      closed[current] = 1;
      const node = this.nodes[current];
      for (const edgeId of node.edges) {
        const edge = this.edges[edgeId];
        const next = edge.a === current ? edge.b : edge.a;
        if (closed[next]) continue;
        const cost = weight === 'time' ? edge.length / edge.speedLimit : edge.length;
        const tentative = g[current] + cost;
        if (tentative < g[next]) {
          g[next] = tentative;
          parent[next] = current;
          heap.push(next, tentative + this.heuristic(next, target, weight));
        }
      }
    }
    if (parent[to] < 0) return null;
    const path: number[] = [];
    for (let c = to; c !== -1; c = parent[c]) path.push(c);
    path.reverse();
    return path;
  }

  private heuristic(nodeId: number, target: Vector3, weight: 'distance' | 'time'): number {
    const d = this.nodes[nodeId].position.distanceTo(target);
    return weight === 'time' ? d / 30 : d;
  }

  /** World-space route between two positions via their nearest nodes. */
  routeBetween(from: V3, to: V3): Vector3[] {
    const a = this.nearestNode(from);
    const b = this.nearestNode(to);
    const ids = this.astar(a.id, b.id);
    if (!ids) return [];
    return ids.map((id) => this.nodes[id].position.clone());
  }

  /** True when every node can reach node 0. */
  isConnected(): boolean {
    if (this.nodes.length === 0) return true;
    const seen = new Uint8Array(this.nodes.length);
    const stack = [0];
    seen[0] = 1;
    let count = 1;
    while (stack.length) {
      const id = stack.pop() as number;
      for (const eid of this.nodes[id].edges) {
        const e = this.edges[eid];
        const next = e.a === id ? e.b : e.a;
        if (!seen[next]) {
          seen[next] = 1;
          count++;
          stack.push(next);
        }
      }
    }
    return count === this.nodes.length;
  }
}

// --- Internals ------------------------------------------------------------------

function pairKey(a: number, b: number): number {
  return a < b ? a * 65536 + b : b * 65536 + a;
}

function collectSegments(polylines: Vector3[][]): Segment[] {
  const out: Segment[] = [];
  for (let p = 0; p < polylines.length; p++) {
    const pts = polylines[p];
    for (let i = 0; i + 1 < pts.length; i++) {
      out.push({ polyline: p, index: i, ax: pts[i].x, az: pts[i].z, ay: pts[i].y, bx: pts[i + 1].x, bz: pts[i + 1].z, by: pts[i + 1].y });
    }
  }
  return out;
}

interface NearestHit {
  index: number;
  point: Vector3;
  direction: Vector3;
  t: number;
}

/** Uniform grid of segment indices; each segment is inserted in every cell it crosses. */
class SegmentGrid {
  private readonly cells = new Map<number, number[]>();
  private stamp: Int32Array = new Int32Array(0);
  private stampValue = 1;

  constructor(private readonly cellSize: number) {}

  private key(cx: number, cz: number): number {
    return (cx + 32768) * 65536 + (cz + 32768);
  }

  build(segments: Segment[]): void {
    this.cells.clear();
    this.stamp = new Int32Array(segments.length);
    const cs = this.cellSize;
    for (let i = 0; i < segments.length; i++) {
      const s = segments[i];
      const len = Math.hypot(s.bx - s.ax, s.bz - s.az);
      const steps = Math.max(1, Math.ceil(len / (cs * 0.5)));
      let last = -1;
      for (let k = 0; k <= steps; k++) {
        const t = k / steps;
        const x = s.ax + (s.bx - s.ax) * t;
        const z = s.az + (s.bz - s.az) * t;
        const key = this.key(Math.floor(x / cs), Math.floor(z / cs));
        if (key === last) continue;
        last = key;
        let bucket = this.cells.get(key);
        if (!bucket) {
          bucket = [];
          this.cells.set(key, bucket);
        }
        if (bucket[bucket.length - 1] !== i) bucket.push(i);
      }
    }
  }

  nearest(x: number, z: number, segments: Segment[]): NearestHit | null {
    if (segments.length === 0) return null;
    for (let radius = this.cellSize; radius <= 4000; radius *= 2) {
      const hit = this.nearestWithin(x, z, radius, segments);
      if (hit) return hit;
    }
    return null;
  }

  private nearestWithin(x: number, z: number, radius: number, segments: Segment[]): NearestHit | null {
    const cs = this.cellSize;
    const cx0 = Math.floor((x - radius) / cs);
    const cx1 = Math.floor((x + radius) / cs);
    const cz0 = Math.floor((z - radius) / cs);
    const cz1 = Math.floor((z + radius) / cs);
    this.stampValue++;
    let bestD = radius * radius;
    let best = -1;
    let bestT = 0;
    for (let cx = cx0; cx <= cx1; cx++) {
      for (let cz = cz0; cz <= cz1; cz++) {
        const bucket = this.cells.get(this.key(cx, cz));
        if (!bucket) continue;
        for (const i of bucket) {
          if (this.stamp[i] === this.stampValue) continue;
          this.stamp[i] = this.stampValue;
          const s = segments[i];
          const t = projectT(s, x, z);
          const px = s.ax + (s.bx - s.ax) * t;
          const pz = s.az + (s.bz - s.az) * t;
          const d = (px - x) * (px - x) + (pz - z) * (pz - z);
          if (d < bestD) {
            bestD = d;
            best = i;
            bestT = t;
          }
        }
      }
    }
    if (best < 0) return null;
    return makeHit(segments[best], best, bestT);
  }

  randomInRing(
    x: number,
    z: number,
    minR: number,
    maxR: number,
    segments: Segment[],
    rng: Random,
    accept: (s: Segment) => boolean,
  ): { segment: Segment; point: Vector3; direction: Vector3; t: number } | null {
    const cs = this.cellSize;
    const cx0 = Math.floor((x - maxR) / cs);
    const cx1 = Math.floor((x + maxR) / cs);
    const cz0 = Math.floor((z - maxR) / cs);
    const cz1 = Math.floor((z + maxR) / cs);
    this.stampValue++;
    const candidates: number[] = [];
    for (let cx = cx0; cx <= cx1; cx++) {
      for (let cz = cz0; cz <= cz1; cz++) {
        const bucket = this.cells.get(this.key(cx, cz));
        if (!bucket) continue;
        for (const i of bucket) {
          if (this.stamp[i] === this.stampValue) continue;
          this.stamp[i] = this.stampValue;
          if (accept(segments[i])) candidates.push(i);
        }
      }
    }
    // Try random points on random candidate segments until one falls inside the ring.
    for (let attempt = 0; attempt < 40 && candidates.length > 0; attempt++) {
      const i = candidates[Math.floor(rng.next() * candidates.length)];
      const s = segments[i];
      const t = rng.next();
      const px = s.ax + (s.bx - s.ax) * t;
      const pz = s.az + (s.bz - s.az) * t;
      const d = Math.hypot(px - x, pz - z);
      if (d >= minR && d <= maxR) {
        const hit = makeHit(s, i, t);
        return { segment: s, point: hit.point, direction: hit.direction, t };
      }
    }
    return null;
  }
}

function projectT(s: Segment, x: number, z: number): number {
  const dx = s.bx - s.ax;
  const dz = s.bz - s.az;
  const len2 = dx * dx + dz * dz;
  if (len2 < 1e-9) return 0;
  const t = ((x - s.ax) * dx + (z - s.az) * dz) / len2;
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

function makeHit(s: Segment, index: number, t: number): NearestHit {
  const point = new Vector3(s.ax + (s.bx - s.ax) * t, s.ay + (s.by - s.ay) * t, s.az + (s.bz - s.az) * t);
  const direction = new Vector3(s.bx - s.ax, s.by - s.ay, s.bz - s.az).normalize();
  return { index, point, direction, t };
}

/** Binary min-heap keyed by priority, storing node ids. */
class MinHeap {
  private ids: number[] = [];
  private keys: number[] = [];

  get size(): number {
    return this.ids.length;
  }

  push(id: number, key: number): void {
    this.ids.push(id);
    this.keys.push(key);
    let i = this.ids.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.keys[p] <= this.keys[i]) break;
      this.swap(i, p);
      i = p;
    }
  }

  pop(): number {
    const top = this.ids[0];
    const lastId = this.ids.pop() as number;
    const lastKey = this.keys.pop() as number;
    if (this.ids.length > 0) {
      this.ids[0] = lastId;
      this.keys[0] = lastKey;
      let i = 0;
      const n = this.ids.length;
      for (;;) {
        const l = i * 2 + 1;
        const r = l + 1;
        let m = i;
        if (l < n && this.keys[l] < this.keys[m]) m = l;
        if (r < n && this.keys[r] < this.keys[m]) m = r;
        if (m === i) break;
        this.swap(i, m);
        i = m;
      }
    }
    return top;
  }

  private swap(i: number, j: number): void {
    const ti = this.ids[i];
    this.ids[i] = this.ids[j];
    this.ids[j] = ti;
    const tk = this.keys[i];
    this.keys[i] = this.keys[j];
    this.keys[j] = tk;
  }
}
