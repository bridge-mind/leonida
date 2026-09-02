import { Vector3 } from 'three';
import type { V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { Blip } from '../../core/GameState';
import type { INavigationPeer } from '../../core/peers';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { NavConfig as N } from './config';

/**
 * Owns `game.state.blips`, `game.state.waypoint` and `game.state.route`.
 * Landmarks become blips at init; the waypoint route is an A* walk over the road
 * graph that is re-planned when the player strays from it.
 */
export class NavigationSystem implements GameSystem, INavigationPeer {
  private readonly blipMap = new Map<string, Blip>();
  private readonly waypointPos: V3 = { x: 0, y: 0, z: 0 };
  private readonly waypointBlip: Blip = { id: 'waypoint', kind: 'waypoint', position: this.waypointPos, label: 'Waypoint' };
  private hasWaypoint = false;
  /** Route points are recycled so re-planning never allocates in steady state. */
  private readonly pointPool: V3[] = [];
  private strayTimer = 0;
  private routeLength = 0;
  private distanceAlong = 0;

  constructor(private readonly game: Game) {}

  init(): void {
    if (this.game.systems.has('world')) {
      for (const l of this.game.world.landmarks.list) {
        this.blipMap.set(l.id, { id: l.id, kind: l.kind, position: l.door, label: l.name });
      }
    }
    this.rebuildBlips();
  }

  // --- INavigationPeer -----------------------------------------------------------

  setWaypoint(pos: V3 | null): void {
    const s = this.game.state;
    if (!pos) {
      if (!this.hasWaypoint) return;
      this.hasWaypoint = false;
      s.waypoint = null;
      s.route.length = 0;
      this.routeLength = 0;
      this.blipMap.delete('waypoint');
      this.rebuildBlips();
      return;
    }
    this.waypointPos.x = pos.x;
    this.waypointPos.y = pos.y;
    this.waypointPos.z = pos.z;
    this.hasWaypoint = true;
    s.waypoint = this.waypointPos;
    this.blipMap.set('waypoint', this.waypointBlip);
    this.rebuildBlips();
    this.replan();
  }

  get waypoint(): V3 | null {
    return this.hasWaypoint ? this.waypointPos : null;
  }

  addBlip(blip: Blip): void {
    this.blipMap.set(blip.id, blip);
    this.rebuildBlips();
  }

  removeBlip(id: string): void {
    if (!this.blipMap.delete(id)) return;
    this.rebuildBlips();
  }

  routeTo(pos: V3): Vector3[] {
    if (!this.game.systems.has('world') || !this.game.systems.has('player')) return [];
    const pts = this.game.world.roadGraph.routeBetween(this.game.player.position, pos);
    if (pts.length === 0) return pts;
    pts.push(new Vector3(pos.x, pos.y, pos.z));
    return pts;
  }

  /** Meters left along the route (straight-line when there is no road route). */
  get distanceToWaypoint(): number {
    return this.hasWaypoint ? this.distanceAlong : 0;
  }

  // --- Loop ----------------------------------------------------------------------

  update(): void {
    if (!this.hasWaypoint || !this.game.systems.has('player')) return;
    this.strayTimer -= this.game.time.realDt;
    if (this.strayTimer > 0) return;
    this.strayTimer = N.strayCheckInterval;
    const p = this.game.player.position;
    const dx = p.x - this.waypointPos.x;
    const dz = p.z - this.waypointPos.z;
    if (dx * dx + dz * dz < N.arriveDistance * N.arriveDistance) {
      this.setWaypoint(null);
      return;
    }
    this.measureRoute(p);
  }

  // --- Internals -----------------------------------------------------------------

  private rebuildBlips(): void {
    const out = this.game.state.blips;
    out.length = 0;
    for (const b of this.blipMap.values()) out.push(b);
  }

  private point(i: number, x: number, y: number, z: number): V3 {
    let p = this.pointPool[i];
    if (!p) {
      p = { x: 0, y: 0, z: 0 };
      this.pointPool[i] = p;
    }
    p.x = x;
    p.y = y;
    p.z = z;
    return p;
  }

  private replan(): void {
    const route = this.game.state.route;
    route.length = 0;
    this.routeLength = 0;
    this.strayTimer = N.strayCheckInterval;
    if (!this.game.systems.has('world') || !this.game.systems.has('player')) return;
    const graph = this.game.world.roadGraph;
    const from = this.game.player.position;
    const target = this.waypointPos;
    const ids = graph.astar(graph.nearestNode(from).id, graph.nearestNode(target).id);
    if (!ids) {
      this.distanceAlong = Math.hypot(target.x - from.x, target.z - from.z);
      return;
    }
    const nodes = graph.nodes;
    let start = 0;
    let end = ids.length;
    // Skip the first node when the player already stands along the first edge, so the
    // line does not double back to an intersection behind them.
    if (ids.length >= 2 && projectT(from, nodes[ids[0]].position, nodes[ids[1]].position) > 0.15) start = 1;
    // Same at the far end: stop at the target instead of overshooting to the next node.
    if (end - start >= 2 && projectT(target, nodes[ids[end - 1]].position, nodes[ids[end - 2]].position) > 0.15) end--;
    let n = 0;
    for (let i = start; i < end; i++) {
      const pos = nodes[ids[i]].position;
      route.push(this.point(n++, pos.x, N.routeY, pos.z));
    }
    const last = route[route.length - 1];
    if (last && Math.hypot(last.x - target.x, last.z - target.z) < N.finalPointMergeDistance) {
      route.pop();
      n--;
    }
    route.push(this.point(n++, target.x, target.y, target.z));
    for (let i = 1; i < route.length; i++) this.routeLength += Math.hypot(route[i].x - route[i - 1].x, route[i].z - route[i - 1].z);
    this.measureRoute(from);
  }

  /** Nearest route segment to the player; re-plans when they strayed, else updates the distance left. */
  private measureRoute(p: V3): void {
    const route = this.game.state.route;
    if (route.length === 0) {
      this.replan();
      return;
    }
    if (route.length === 1) {
      this.distanceAlong = Math.hypot(route[0].x - p.x, route[0].z - p.z);
      return;
    }
    let bestD = Infinity;
    let bestIdx = 0;
    let bestT = 0;
    for (let i = 0; i + 1 < route.length; i++) {
      const a = route[i];
      const b = route[i + 1];
      const t = projectT(p, a, b);
      const px = a.x + (b.x - a.x) * t;
      const pz = a.z + (b.z - a.z) * t;
      const d = (px - p.x) * (px - p.x) + (pz - p.z) * (pz - p.z);
      if (d < bestD) {
        bestD = d;
        bestIdx = i;
        bestT = t;
      }
    }
    if (bestD > N.strayDistance * N.strayDistance) {
      this.replan();
      return;
    }
    let remaining = Math.sqrt(bestD);
    const a = route[bestIdx];
    const b = route[bestIdx + 1];
    remaining += Math.hypot(b.x - a.x, b.z - a.z) * (1 - bestT);
    for (let i = bestIdx + 1; i + 1 < route.length; i++) {
      remaining += Math.hypot(route[i + 1].x - route[i].x, route[i + 1].z - route[i].z);
    }
    this.distanceAlong = remaining;
  }

  serialize(): unknown {
    return { waypoint: this.hasWaypoint ? { ...this.waypointPos } : null };
  }

  deserialize(data: unknown): void {
    const d = data as { waypoint?: V3 | null } | null;
    if (!d) return;
    if (d.waypoint && typeof d.waypoint.x === 'number') this.setWaypoint(d.waypoint);
    else this.setWaypoint(null);
  }

  dispose(): void {
    this.setWaypoint(null);
    this.blipMap.clear();
    this.rebuildBlips();
  }

  debugActions = {
    waypoint: (x: unknown, z: unknown): number => {
      this.setWaypoint({ x: Number(x), y: 0, z: Number(z) });
      return this.game.state.route.length;
    },
    clear: (): void => this.setWaypoint(null),
    route: (): { points: number; length: number; distance: number } => ({
      points: this.game.state.route.length,
      length: Math.round(this.routeLength),
      distance: Math.round(this.distanceAlong),
    }),
    blips: (): string[] => this.game.state.blips.map((b) => `${b.kind}:${b.id}`),
  };
}

/** Parameter of `p` projected onto segment a→b, clamped to [0, 1]. */
function projectT(p: V3, a: V3, b: V3): number {
  const dx = b.x - a.x;
  const dz = b.z - a.z;
  const len2 = dx * dx + dz * dz;
  if (len2 < 1e-9) return 0;
  const t = ((p.x - a.x) * dx + (p.z - a.z) * dz) / len2;
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

const module: SystemModule = {
  id: 'navigation',
  priority: PRIORITY.navigation,
  create: (game) => new NavigationSystem(game),
};

export default module;
