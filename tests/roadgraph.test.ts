import { describe, expect, it } from 'vitest';
import { Random } from '../src/core/Random';
import { LANDMARKS } from '../src/data/landmarks';
import { buildRoadGraph } from '../src/world/RoadGraphBuilder';
import { ROAD_LINES, type RoadLineDef } from '../src/world/roadNetwork';

describe('RoadGraph builder on a small grid', () => {
  const lines: RoadLineDef[] = [
    { axis: 'x', at: 0, from: 0, to: 200, kind: 'street' },
    { axis: 'x', at: 100, from: 0, to: 200, kind: 'avenue' },
    { axis: 'z', at: 0, from: 0, to: 100, kind: 'street' },
    { axis: 'z', at: 100, from: 0, to: 100, kind: 'street' },
    { axis: 'z', at: 200, from: 0, to: 100, kind: 'street' },
  ];
  const graph = buildRoadGraph(lines);

  it('creates a node at every intersection and edges between consecutive ones', () => {
    expect(graph.nodes.length).toBe(6);
    expect(graph.edges.length).toBe(7);
    const corner = graph.nodes.find((n) => n.position.x === 0 && n.position.z === 0);
    expect(corner?.edges.length).toBe(2);
    const t = graph.nodes.find((n) => n.position.x === 100 && n.position.z === 0);
    expect(t?.edges.length).toBe(3);
    expect(t?.hasLight).toBe(true);
    expect(corner?.hasLight).toBe(false);
  });

  it('derives right-hand lanes and sidewalks', () => {
    const avenue = graph.edges.find((e) => e.kind === 'avenue');
    expect(avenue).toBeDefined();
    expect(avenue?.lanes.length).toBe(4);
    expect(avenue?.sidewalks.length).toBe(2);
    const street = graph.edges.find((e) => e.kind === 'street');
    expect(street?.lanes.length).toBe(2);
    // Lane 0 traveling a→b lies on the right of the direction of travel.
    const lane = graph.lanes[street!.lanes[0]];
    const mid = lane.points[0].clone().add(lane.points[1]).multiplyScalar(0.5);
    const a = graph.nodes[street!.a].position;
    const rel = mid.clone().sub(a);
    expect(rel.dot(street!.right)).toBeGreaterThan(0);
  });

  it('finds shortest paths with A*', () => {
    const from = graph.nodes.find((n) => n.position.x === 0 && n.position.z === 0)!.id;
    const to = graph.nodes.find((n) => n.position.x === 200 && n.position.z === 100)!.id;
    const path = graph.astar(from, to);
    expect(path).not.toBeNull();
    expect(path![0]).toBe(from);
    expect(path![path!.length - 1]).toBe(to);
    let length = 0;
    for (let i = 0; i + 1 < path!.length; i++) length += graph.edgeBetween(path![i], path![i + 1])!.length;
    expect(length).toBeCloseTo(300, 5);
  });

  it('answers nearest queries', () => {
    const lane = graph.nearestLanePoint({ x: 50, y: 0, z: 2 });
    expect(lane).not.toBeNull();
    expect(Math.abs(lane!.point.z)).toBeLessThan(4);
    const sidewalk = graph.nearestSidewalkPoint({ x: 50, y: 0, z: 9 });
    expect(sidewalk).not.toBeNull();
    expect(sidewalk!.point.z).toBeCloseTo(8, 5);
    const rng = new Random(7);
    const random = graph.randomLanePointNear({ x: 100, y: 0, z: 50 }, 10, 80, rng);
    expect(random).not.toBeNull();
    const d = Math.hypot(random!.point.x - 100, random!.point.z - 50);
    expect(d).toBeGreaterThanOrEqual(10);
    expect(d).toBeLessThanOrEqual(80);
  });
});

describe('Vice City road network', () => {
  const graph = buildRoadGraph(ROAD_LINES);

  it('is one connected component with lights at intersections', () => {
    expect(graph.nodes.length).toBeGreaterThan(150);
    expect(graph.isConnected()).toBe(true);
    expect(graph.nodes.filter((n) => n.hasLight).length).toBeGreaterThan(80);
  });

  it('routes between 50 random node pairs', () => {
    const rng = new Random(42);
    for (let i = 0; i < 50; i++) {
      const a = rng.int(0, graph.nodes.length - 1);
      const b = rng.int(0, graph.nodes.length - 1);
      const path = graph.astar(a, b);
      expect(path).not.toBeNull();
      expect(path![0]).toBe(a);
      expect(path![path!.length - 1]).toBe(b);
    }
  });

  it('has elevated causeway nodes and flat island nodes', () => {
    const deck = graph.nodes.filter((n) => n.position.y > 7);
    expect(deck.length).toBeGreaterThanOrEqual(4);
    const island = graph.nodes.find((n) => n.position.x === 560 && n.position.z === 0);
    expect(island?.position.y).toBe(0);
  });

  it('places every landmark door within 15 m of a node', () => {
    for (const l of LANDMARKS) {
      const node = graph.nearestNode(l.door);
      const d = Math.hypot(node.position.x - l.door.x, node.position.z - l.door.z);
      expect(d, l.id).toBeLessThanOrEqual(15);
    }
  });

  it('generates parking spots and crosswalks', () => {
    expect(graph.parkingSpots.length).toBeGreaterThan(500);
    expect(graph.crosswalks.length).toBeGreaterThan(200);
    for (const spot of graph.parkingSpots.slice(0, 50)) {
      const edge = graph.edges[spot.edgeId];
      expect(edge.kind === 'street' || edge.kind === 'avenue').toBe(true);
    }
  });
});
