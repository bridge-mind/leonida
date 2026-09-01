import { Vector3 } from 'three';
import { Config } from '../core/Config';
import { RoadGraph, type LanePolyline, type RoadEdge, type RoadNode, type SidewalkPolyline } from './RoadGraph';
import { ROAD_KINDS, type RoadLineDef } from './roadNetwork';

const SIDEWALK_W = Config.world.sidewalkWidth;
const LANE_W = Config.world.laneWidth;
const PARKING_SPACING = 6.5;
const EPS = 0.01;

/**
 * Builds a RoadGraph from axis-aligned line definitions:
 * nodes at every intersection, line end and profile keyframe; edges between
 * consecutive nodes along a line; then lanes, sidewalks, crosswalks and parking.
 */
export function buildRoadGraph(lines: readonly RoadLineDef[]): RoadGraph {
  const graph = new RoadGraph();
  const nodeKey = new Map<string, number>();

  const getNode = (x: number, z: number, y: number): RoadNode => {
    const key = `${Math.round(x / EPS)},${Math.round(z / EPS)}`;
    let id = nodeKey.get(key);
    if (id === undefined) {
      id = graph.nodes.length;
      nodeKey.set(key, id);
      graph.nodes.push({
        id,
        position: new Vector3(x, y, z),
        edges: [],
        lightPhase: (Math.round(x / 100) + Math.round(z / 100)) & 1,
        hasLight: false,
        halfNS: 0,
        halfEW: 0,
      });
    } else if (y !== 0) {
      graph.nodes[id].position.y = y;
    }
    return graph.nodes[id];
  };

  for (const line of lines) {
    const lo = Math.min(line.from, line.to);
    const hi = Math.max(line.from, line.to);
    const cuts = new Set<number>([lo, hi]);
    if (line.profile) for (const p of line.profile) if (p.at >= lo - EPS && p.at <= hi + EPS) cuts.add(p.at);
    for (const other of lines) {
      if (other === line || other.axis === line.axis) continue;
      const olo = Math.min(other.from, other.to);
      const ohi = Math.max(other.from, other.to);
      if (other.at >= lo - EPS && other.at <= hi + EPS && line.at >= olo - EPS && line.at <= ohi + EPS) cuts.add(other.at);
    }
    const sorted = [...cuts].sort((a, b) => a - b);
    const def = ROAD_KINDS[line.kind];
    let prev: RoadNode | null = null;
    for (const c of sorted) {
      const y = elevationAt(line, c);
      const node = line.axis === 'x' ? getNode(c, line.at, y) : getNode(line.at, c, y);
      if (prev && prev !== node && !hasEdge(graph, prev.id, node.id)) addEdge(graph, prev, node, line, def);
      prev = node;
    }
  }

  computeNodeMetrics(graph);
  for (const edge of graph.edges) deriveEdge(graph, edge);
  graph.finalize();
  return graph;
}

function hasEdge(graph: RoadGraph, a: number, b: number): boolean {
  for (const eid of graph.nodes[a].edges) {
    const e = graph.edges[eid];
    if ((e.a === a && e.b === b) || (e.a === b && e.b === a)) return true;
  }
  return false;
}

function elevationAt(line: RoadLineDef, at: number): number {
  const prof = line.profile;
  if (!prof || prof.length === 0) return 0;
  if (at <= prof[0].at) return prof[0].y;
  for (let i = 0; i + 1 < prof.length; i++) {
    if (at >= prof[i].at && at <= prof[i + 1].at) {
      const t = (at - prof[i].at) / Math.max(1e-6, prof[i + 1].at - prof[i].at);
      return prof[i].y + (prof[i + 1].y - prof[i].y) * t;
    }
  }
  return prof[prof.length - 1].y;
}

function addEdge(graph: RoadGraph, a: RoadNode, b: RoadNode, line: RoadLineDef, def: (typeof ROAD_KINDS)[keyof typeof ROAD_KINDS]): void {
  const direction = new Vector3().subVectors(b.position, a.position);
  direction.y = 0;
  const length = a.position.distanceTo(b.position);
  direction.normalize();
  const edge: RoadEdge = {
    id: graph.edges.length,
    a: a.id,
    b: b.id,
    kind: line.kind,
    lanesPerDirection: def.lanesPerDirection,
    width: def.width,
    speedLimit: def.speedLimit,
    isHighway: def.isHighway,
    surface: def.surface,
    length,
    direction,
    right: new Vector3(-direction.z, 0, direction.x),
    lanes: [],
    sidewalks: [],
    trimA: 0,
    trimB: 0,
  };
  graph.edges.push(edge);
  a.edges.push(edge.id);
  b.edges.push(edge.id);
}

function isNorthSouth(edge: RoadEdge): boolean {
  return Math.abs(edge.direction.z) > Math.abs(edge.direction.x);
}

function computeNodeMetrics(graph: RoadGraph): void {
  for (const node of graph.nodes) {
    let halfNS = 0;
    let halfEW = 0;
    for (const eid of node.edges) {
      const e = graph.edges[eid];
      if (isNorthSouth(e)) halfNS = Math.max(halfNS, e.width / 2);
      else halfEW = Math.max(halfEW, e.width / 2);
    }
    node.halfNS = halfNS;
    node.halfEW = halfEW;
    node.hasLight = node.edges.length >= 3;
  }
  for (const edge of graph.edges) {
    const ns = isNorthSouth(edge);
    const a = graph.nodes[edge.a];
    const b = graph.nodes[edge.b];
    // Trim by the perpendicular road's half width so the edge butts against the intersection patch.
    edge.trimA = ns ? a.halfEW : a.halfNS;
    edge.trimB = ns ? b.halfEW : b.halfNS;
    const usable = edge.length - edge.trimA - edge.trimB;
    if (usable < 2) {
      const shrink = (2 - usable) / 2;
      edge.trimA = Math.max(0, edge.trimA - shrink);
      edge.trimB = Math.max(0, edge.trimB - shrink);
    }
  }
}

function pointAlong(graph: RoadGraph, edge: RoadEdge, dist: number, lateral: number, out: Vector3): Vector3 {
  const a = graph.nodes[edge.a].position;
  const b = graph.nodes[edge.b].position;
  const t = edge.length > 0 ? dist / edge.length : 0;
  out.set(a.x + edge.direction.x * dist, a.y + (b.y - a.y) * t, a.z + edge.direction.z * dist);
  out.x += edge.right.x * lateral;
  out.z += edge.right.z * lateral;
  return out;
}

function deriveEdge(graph: RoadGraph, edge: RoadEdge): void {
  const def = ROAD_KINDS[edge.kind];
  const start = edge.trimA;
  const end = edge.length - edge.trimB;
  const half = edge.width / 2;

  // Lanes: right-hand traffic. a→b lanes on the +right side, b→a on the -right side.
  for (let dir = 0; dir < 2; dir++) {
    for (let i = 0; i < def.lanesPerDirection; i++) {
      const lateral = (dir === 0 ? 1 : -1) * LANE_W * (i + 0.5);
      const p0 = pointAlong(graph, edge, start, lateral, new Vector3());
      const p1 = pointAlong(graph, edge, end, lateral, new Vector3());
      const lane: LanePolyline = {
        id: graph.lanes.length,
        edgeId: edge.id,
        from: dir === 0 ? edge.a : edge.b,
        to: dir === 0 ? edge.b : edge.a,
        laneIndex: i,
        points: dir === 0 ? [p0, p1] : [p1, p0],
      };
      graph.lanes.push(lane);
      edge.lanes.push(lane.id);
    }
  }

  if (def.sidewalks) {
    const sStart = start + SIDEWALK_W;
    const sEnd = end - SIDEWALK_W;
    if (sEnd - sStart > 1) {
      for (const side of [1, -1] as const) {
        const lateral = side * (half + SIDEWALK_W / 2);
        const sw: SidewalkPolyline = {
          id: graph.sidewalks.length,
          edgeId: edge.id,
          side,
          points: [pointAlong(graph, edge, sStart, lateral, new Vector3()), pointAlong(graph, edge, sEnd, lateral, new Vector3())],
        };
        graph.sidewalks.push(sw);
        edge.sidewalks.push(sw.id);
      }
    }
    // Crosswalks across this edge just outside each intersection.
    for (const [nodeId, dist] of [
      [edge.a, start + SIDEWALK_W / 2],
      [edge.b, end - SIDEWALK_W / 2],
    ] as const) {
      const node = graph.nodes[nodeId];
      if (node.edges.length < 3) continue;
      graph.crosswalks.push({
        nodeId,
        edgeId: edge.id,
        a: pointAlong(graph, edge, dist, half + SIDEWALK_W / 2, new Vector3()),
        b: pointAlong(graph, edge, dist, -(half + SIDEWALK_W / 2), new Vector3()),
      });
    }
  }

  if (def.parking) {
    const parkingLateral = half - 1.3;
    const usable = end - start - 2 * SIDEWALK_W - 6;
    const count = Math.floor(usable / PARKING_SPACING);
    const heading = Math.atan2(edge.direction.x, edge.direction.z);
    for (let i = 0; i < count; i++) {
      const d = start + SIDEWALK_W + 4 + i * PARKING_SPACING + PARKING_SPACING / 2;
      for (const side of [1, -1] as const) {
        graph.parkingSpots.push({
          position: pointAlong(graph, edge, d, side * parkingLateral, new Vector3()),
          heading: side === 1 ? heading : heading + Math.PI,
          edgeId: edge.id,
          side,
          kind: 'curb',
        });
      }
    }
  }
}
