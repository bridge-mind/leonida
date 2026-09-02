import { Vector3 } from 'three';
import type { IVehicle, V3, VehicleDriver } from '../../core/entities';
import type { Random } from '../../core/Random';
import type { LanePoint, LanePolyline, RoadEdge, RoadGraph, RoadNode } from '../../world/RoadGraph';
import { wrapAngle } from '../vehicles/handling';
import { TrafficConfig } from './config';
import { edgeAxis, type SignalState, type TrafficLights } from './TrafficLights';

const D = TrafficConfig.driver;

/** What a driver needs from the traffic system (implemented by TrafficSystem). */
export interface TrafficContext {
  readonly graph: RoadGraph;
  readonly lights: TrafficLights;
  readonly rng: Random;
  /** Vehicles within `radius` of `pos` (shared scratch array; consume before the next call). */
  queryVehicles(pos: V3, radius: number): readonly IVehicle[];
  /** Player position when on foot, else null. */
  playerOnFoot(): V3 | null;
  honk(vehicle: IVehicle): void;
}

const aim = new Vector3();
const signal: SignalState = { green: null, yellow: false, timeToRed: Infinity };

/**
 * Lane follower: tracks its lane polyline with a lookahead point, picks a random
 * outgoing edge at each node, keeps a gap behind the car ahead, obeys the light
 * clock, honks when blocked and reroutes when stuck.
 */
export class TrafficDriver implements VehicleDriver {
  lane: LanePolyline;
  edge: RoadEdge;
  private seg = 0;
  private nextLane: LanePolyline | null = null;
  private committed = false;
  private readonly speedVariance: number;
  blocked = 0;
  stuck = 0;
  /** Seconds sitting still with a green light and a clear road: should stay near zero. */
  idle = 0;
  private honkCooldown = 0;
  /** Set once the car has been stuck long enough that the spawner should recycle it. */
  wantsDespawn = false;
  /** True while braking for a red light (not counted as blocked). */
  stoppedForLight = false;

  constructor(
    private readonly ctx: TrafficContext,
    start: LanePoint,
  ) {
    this.lane = start.lane;
    this.edge = start.edge;
    this.seg = start.segment;
    this.speedVariance = ctx.rng.range(D.speedVarianceMin, D.speedVarianceMax);
  }

  update(v: IVehicle, dt: number): void {
    const pos = v.position;
    const speed = v.state.speedMs;
    this.honkCooldown -= dt;

    // --- Progress along the lane -------------------------------------------------
    let t = this.advanceSegments(pos);
    if (t >= 1 && this.seg >= this.lane.points.length - 2) {
      this.lane = this.nextLane ?? this.chooseNext();
      this.edge = this.ctx.graph.edges[this.lane.edgeId];
      this.nextLane = null;
      this.committed = false;
      this.seg = 0;
      t = this.advanceSegments(pos);
    }
    const distToEnd = this.distanceToEnd(t);
    if (distToEnd < 40 && !this.nextLane) this.nextLane = this.chooseNext();

    // --- Steering toward a lookahead point ---------------------------------------
    const lookahead = Math.max(D.lookaheadMin, speed * D.lookaheadFactor);
    this.pointAhead(t, lookahead, aim);
    const desired = Math.atan2(aim.x - pos.x, aim.z - pos.z);
    const err = wrapAngle(desired - v.heading);
    // A positive heading error is a left turn, which is negative steer.
    v.controls.steer = clamp(-err * D.steerGain, -1, 1);

    // --- Speed planning ------------------------------------------------------------
    let target = clamp(this.edge.speedLimit * this.speedVariance, D.speedFloor, D.speedCeiling);
    if (this.nextLane && distToEnd < D.turnSlowDistance) {
      const turn = Math.abs(wrapAngle(laneHeading(this.nextLane) - laneHeading(this.lane, this.lane.points.length - 2)));
      if (turn > D.turnAngle) target = Math.min(target, D.turnSpeed);
    }
    let allowed = target;
    let obstacleStop = false;

    const gapAllowed = this.followingSpeed(v);
    if (gapAllowed < allowed) {
      allowed = gapAllowed;
      obstacleStop = gapAllowed < 0.5;
    }

    this.stoppedForLight = false;
    const node = this.ctx.graph.nodes[this.lane.to];
    if (node.hasLight && !this.committed && t < 1) {
      this.ctx.lights.stateFor(node.id, this.edge.id, signal);
      const axis = edgeAxis(this.edge);
      const red = signal.green !== axis;
      const yellowStop = signal.yellow && distToEnd > speed * D.yellowStopFactor + D.yellowStopMargin;
      if (red || yellowStop) {
        if (distToEnd < D.commitDistance && speed > D.commitSpeed) {
          this.committed = true;
        } else {
          const gap = distToEnd - D.stopLineOffset;
          const lightAllowed = gap < 0.3 ? 0 : Math.sqrt(2 * D.planDecel * gap);
          if (lightAllowed < allowed) {
            allowed = lightAllowed;
            this.stoppedForLight = lightAllowed < 0.5;
            if (this.stoppedForLight) obstacleStop = false;
          }
        }
      }
    }

    const dv = allowed - speed;
    const c = v.controls;
    if (dv > 0) {
      c.throttle = Math.min(1, dv * D.throttleGain);
      c.brake = 0;
    } else {
      c.throttle = 0;
      c.brake = Math.min(1, -dv * D.brakeGain);
    }
    if (allowed < 0.3 && speed < 0.6) {
      c.throttle = 0;
      c.brake = 1;
    }
    c.handbrake = false;

    // --- Blocked / stuck handling ------------------------------------------------
    if (obstacleStop && speed < 0.5) {
      this.blocked += dt;
      this.stuck += dt;
      if (this.blocked > D.honkAfter && this.honkCooldown <= 0) {
        this.honkCooldown = D.honkCooldown;
        this.ctx.honk(v);
      }
      if (this.blocked > D.rerouteAfter) {
        this.reroute();
        this.blocked = D.honkAfter;
      }
      if (this.stuck > TrafficConfig.stuckDespawnSeconds) this.wantsDespawn = true;
    } else if (speed > 1) {
      this.blocked = 0;
      this.stuck = 0;
    }
    this.idle = speed < 0.3 && !obstacleStop && !this.stoppedForLight ? this.idle + dt : 0;
  }

  // --- Lane geometry ---------------------------------------------------------------

  /** Projects onto the current segment, stepping to later segments once passed; returns t. */
  private advanceSegments(pos: V3): number {
    const pts = this.lane.points;
    for (;;) {
      const t = projectT(pts[this.seg], pts[this.seg + 1], pos);
      if (t < 1 || this.seg >= pts.length - 2) return t;
      this.seg++;
    }
  }

  private distanceToEnd(t: number): number {
    const pts = this.lane.points;
    let d = (1 - Math.min(1, t)) * pts[this.seg].distanceTo(pts[this.seg + 1]);
    for (let i = this.seg + 1; i + 1 < pts.length; i++) d += pts[i].distanceTo(pts[i + 1]);
    return d;
  }

  /** Point `dist` ahead of the projection, spilling into the next lane past the end. */
  private pointAhead(t: number, dist: number, out: Vector3): void {
    const pts = this.lane.points;
    let remaining = dist;
    let i = this.seg;
    let from = Math.max(0, t);
    while (i + 1 < pts.length) {
      const a = pts[i];
      const b = pts[i + 1];
      const len = a.distanceTo(b);
      const left = (1 - from) * len;
      if (remaining <= left) {
        const u = from + remaining / Math.max(1e-6, len);
        out.lerpVectors(a, b, u);
        return;
      }
      remaining -= left;
      from = 0;
      i++;
    }
    const next = this.nextLane ?? (this.nextLane = this.chooseNext());
    const np = next.points;
    const nlen = np[0].distanceTo(np[1]);
    out.lerpVectors(np[0], np[1], Math.min(1, remaining / Math.max(1e-6, nlen)));
  }

  /** Random outgoing lane at the end node; avoids U-turns and dirt roads unless forced. */
  private chooseNext(avoidEdge = -1): LanePolyline {
    const graph = this.ctx.graph;
    const node: RoadNode = graph.nodes[this.lane.to];
    let candidates = node.edges.filter((id) => id !== this.edge.id && id !== avoidEdge && graph.edges[id].kind !== 'dirt');
    if (candidates.length === 0) candidates = node.edges.filter((id) => id !== this.edge.id);
    if (candidates.length === 0) candidates = node.edges;
    const edge = graph.edges[this.ctx.rng.pick(candidates)];
    let best: LanePolyline | null = null;
    for (const laneId of edge.lanes) {
      const lane = graph.lanes[laneId];
      if (lane.from !== node.id) continue;
      if (!best || lane.laneIndex === this.lane.laneIndex) best = lane;
    }
    return best ?? graph.lanes[edge.lanes[0]];
  }

  /** Blocked too long: change lane on multi-lane roads, otherwise pick another turn. */
  private reroute(): void {
    const graph = this.ctx.graph;
    if (this.edge.lanesPerDirection > 1) {
      for (const laneId of this.edge.lanes) {
        const lane = graph.lanes[laneId];
        if (lane.from === this.lane.from && lane.laneIndex !== this.lane.laneIndex) {
          this.lane = lane;
          this.seg = 0;
          this.nextLane = null;
          return;
        }
      }
    }
    const avoid = this.nextLane ? this.nextLane.edgeId : -1;
    this.nextLane = this.chooseNext(avoid);
  }

  // --- Car following ---------------------------------------------------------------

  /** Highest speed that keeps a safe gap to whatever is ahead in the lane corridor. */
  private followingSpeed(v: IVehicle): number {
    const pos = v.position;
    const fx = Math.sin(v.heading);
    const fz = Math.cos(v.heading);
    const rx = fz;
    const rz = -fx;
    const halfLen = v.length / 2;
    let allowed = Infinity;

    aim.set(pos.x + fx * D.probeDistance * 0.5, pos.y, pos.z + fz * D.probeDistance * 0.5);
    const others = this.ctx.queryVehicles(aim, D.probeDistance * 0.5 + 4);
    for (let i = 0; i < others.length; i++) {
      const o = others[i];
      if (o === v) continue;
      const dx = o.position.x - pos.x;
      const dz = o.position.z - pos.z;
      const ahead = dx * fx + dz * fz;
      const lateral = dx * rx + dz * rz;
      if (ahead <= 0 || ahead > D.probeDistance || Math.abs(lateral) > D.probeHalfWidth + o.length * 0.15) continue;
      const gap = ahead - halfLen - o.length / 2;
      const oSpeed = Math.max(0, o.velocity.x * fx + o.velocity.z * fz);
      const a = oSpeed + Math.sqrt(2 * D.planDecel * Math.max(0, gap - D.stopGap));
      if (a < allowed) allowed = gap < D.stopGap ? Math.min(oSpeed, a) : a;
    }

    const ped = this.ctx.playerOnFoot();
    if (ped) {
      const dx = ped.x - pos.x;
      const dz = ped.z - pos.z;
      const ahead = dx * fx + dz * fz;
      const lateral = dx * rx + dz * rz;
      if (ahead > 0 && ahead < D.probeDistance && Math.abs(lateral) < D.pedHalfWidth + 0.4) {
        const gap = ahead - halfLen - 0.4;
        const a = Math.sqrt(2 * D.planDecel * Math.max(0, gap - D.stopGap * 0.75));
        if (a < allowed) allowed = a;
      }
    }
    return allowed;
  }
}

function projectT(a: Vector3, b: Vector3, p: V3): number {
  const dx = b.x - a.x;
  const dz = b.z - a.z;
  const len2 = dx * dx + dz * dz;
  if (len2 < 1e-9) return 1;
  return ((p.x - a.x) * dx + (p.z - a.z) * dz) / len2;
}

function laneHeading(lane: LanePolyline, segment = 0): number {
  const a = lane.points[segment];
  const b = lane.points[segment + 1];
  return Math.atan2(b.x - a.x, b.z - a.z);
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
