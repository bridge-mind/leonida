import { Vector3 } from 'three';
import type { V3 } from '../../../core/entities';
import type { Random } from '../../../core/Random';
import type { RoadGraph, SidewalkPolyline } from '../../../world/RoadGraph';
import { PedConfig } from '../config';

const W = PedConfig.wander;
const tmp = new Vector3();

/**
 * Per-ped sidewalk follower: walks the current polyline toward one end, then hops to a
 * sidewalk whose near endpoint is within `nextSidewalkRadius` of that end (crossing the
 * intersection corner, which is where the crosswalks are). With a threat position the
 * choices prefer whatever leads away from it. Allocation-free after construction.
 */
export class SidewalkNav {
  sidewalk: SidewalkPolyline | null = null;
  /** Index (0 | 1) of the endpoint we are heading for. */
  end = 1;
  readonly target = new Vector3();
  private crossing = false;
  private lateral = 0;

  reset(): void {
    this.sidewalk = null;
    this.crossing = false;
  }

  /** Attach to the nearest sidewalk. Returns false when the graph has none nearby. */
  start(graph: RoadGraph, pos: V3, rng: Random, awayFrom: V3 | null): boolean {
    const hit = graph.nearestSidewalkPoint(pos);
    if (!hit) return false;
    this.sidewalk = hit.sidewalk;
    this.lateral = rng.range(-W.lateralJitter, W.lateralJitter);
    this.crossing = false;
    if (awayFrom) {
      const p = hit.sidewalk.points;
      const d0 = distSq(p[0], awayFrom);
      const d1 = distSq(p[p.length - 1], awayFrom);
      this.end = d1 >= d0 ? 1 : 0;
    } else {
      this.end = rng.chance(0.5) ? 1 : 0;
    }
    this.setEndpointTarget();
    return true;
  }

  reached(pos: V3): boolean {
    const dx = this.target.x - pos.x;
    const dz = this.target.z - pos.z;
    return dx * dx + dz * dz <= W.reachRadius * W.reachRadius;
  }

  /** Called when the target is reached: continue along, or pick the next sidewalk. */
  advance(graph: RoadGraph, rng: Random, awayFrom: V3 | null): void {
    const sw = this.sidewalk;
    if (!sw) return;
    if (this.crossing) {
      this.crossing = false;
      this.setEndpointTarget();
      return;
    }
    const edge = graph.edges[sw.edgeId];
    const nodeId = this.end === 0 ? edge.a : edge.b;
    const node = graph.nodes[nodeId];
    const here = endpoint(sw, this.end);
    let best: SidewalkPolyline | null = null;
    let bestNear = 0;
    let bestScore = -Infinity;
    let seen = 0;
    for (let i = 0; i < node.edges.length; i++) {
      const e = graph.edges[node.edges[i]];
      for (let k = 0; k < e.sidewalks.length; k++) {
        const cand = graph.sidewalks[e.sidewalks[k]];
        if (cand === sw) continue;
        const near = e.a === nodeId ? 0 : 1;
        const np = endpoint(cand, near);
        if (distSq(np, here) > W.nextSidewalkRadius * W.nextSidewalkRadius) continue;
        seen++;
        let score: number;
        if (awayFrom) {
          score = distSq(endpoint(cand, 1 - near), awayFrom) + rng.range(0, 25);
        } else {
          // Reservoir sampling: uniform pick without collecting candidates.
          score = rng.chance(1 / seen) ? 1 : 0;
          if (score === 0) continue;
        }
        if (score > bestScore) {
          bestScore = score;
          best = cand;
          bestNear = near;
        }
      }
    }
    if (!best) {
      // Dead end: turn around.
      this.end = 1 - this.end;
      this.setEndpointTarget();
      return;
    }
    this.sidewalk = best;
    this.end = 1 - bestNear;
    this.lateral = rng.range(-W.lateralJitter, W.lateralJitter);
    const np = endpoint(best, bestNear);
    if (distSq(np, here) > 2.25) {
      this.crossing = true;
      this.jittered(best, np, this.target);
    } else {
      this.setEndpointTarget();
    }
  }

  /** Re-aim along the current sidewalk away from a threat (fleeing peds re-evaluate). */
  faceAway(awayFrom: V3): void {
    const sw = this.sidewalk;
    if (!sw || this.crossing) return;
    const d0 = distSq(sw.points[0], awayFrom);
    const d1 = distSq(sw.points[sw.points.length - 1], awayFrom);
    const wanted = d1 >= d0 ? 1 : 0;
    if (wanted !== this.end) {
      this.end = wanted;
      this.setEndpointTarget();
    }
  }

  private setEndpointTarget(): void {
    const sw = this.sidewalk;
    if (!sw) return;
    this.jittered(sw, endpoint(sw, this.end), this.target);
  }

  private jittered(sw: SidewalkPolyline, point: Vector3, out: Vector3): void {
    const a = sw.points[0];
    const b = sw.points[sw.points.length - 1];
    tmp.set(b.x - a.x, 0, b.z - a.z);
    const len = tmp.length();
    if (len > 1e-6) tmp.multiplyScalar(1 / len);
    out.set(point.x - tmp.z * this.lateral, point.y, point.z + tmp.x * this.lateral);
  }
}

function endpoint(sw: SidewalkPolyline, end: number): Vector3 {
  return end === 0 ? sw.points[0] : sw.points[sw.points.length - 1];
}

function distSq(a: V3, b: V3): number {
  const dx = a.x - b.x;
  const dz = a.z - b.z;
  return dx * dx + dz * dz;
}
