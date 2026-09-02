import { Vector3 } from 'three';
import type { IVehicle, V3, VehicleDriver } from '../../core/entities';
import type { RoadGraph } from '../../world/RoadGraph';
import { wrapAngle } from '../vehicles/handling';
import { PoliceConfig } from './config';
import type { PoliceContext } from './context';

const D = PoliceConfig.driver;
const aim = new Vector3();
const eye = new Vector3();

export type DriverMode = 'route' | 'pursuit' | 'hold' | 'leave';

/**
 * Cruiser AI: A* over the road graph to the player's nearest node (re-pathed every 2 s),
 * lane-offset node following, direct pursuit with ramming once the player's car is close
 * and visible, and a stop-and-bail when the player is on foot nearby. A stuck detector
 * backs the car up for a moment instead of grinding against a wall.
 */
export class PoliceDriver implements VehicleDriver {
  mode: DriverMode = 'route';
  /** Set when the crew should get out; the system performs the exit. */
  wantsBail = false;
  private path: number[] = [];
  private pathIndex = 0;
  private repathTimer = 0;
  private losTimer = 0;
  private hasLos = false;
  private stuckTime = 0;
  private reverseTime = 0;
  private leaveTime = 0;
  private readonly waypoint = new Vector3();

  constructor(
    private readonly ctx: PoliceContext,
    private readonly graph: RoadGraph,
  ) {}

  get leaveSeconds(): number {
    return this.leaveTime;
  }

  standDown(): void {
    if (this.mode !== 'leave') {
      this.mode = 'leave';
      this.leaveTime = 0;
      this.repathTimer = 0;
    }
  }

  update(v: IVehicle, dt: number): void {
    const c = v.controls;
    c.handbrake = false;
    if (this.mode === 'leave') {
      this.leaveTime += dt;
      this.followRoute(v, dt, D.cruiseSpeed * 0.6);
      return;
    }
    if (!this.ctx.playerAlive() || this.ctx.holdFire) {
      this.hold(v);
      return;
    }

    const playerPos = this.ctx.playerPosition();
    const dx = playerPos.x - v.position.x;
    const dz = playerPos.z - v.position.z;
    const dist = Math.sqrt(dx * dx + dz * dz);
    const playerCar = this.ctx.playerVehicle();

    this.losTimer -= dt;
    if (this.losTimer <= 0) {
      this.losTimer = D.losInterval;
      eye.set(v.position.x, v.position.y + 1.75, v.position.z);
      this.hasLos = dist <= D.losRange && this.ctx.sight.toPlayer(eye, D.losRange, v);
      if (this.hasLos) this.ctx.reportSighting('cruiser');
    }

    // Player on foot and close: pull over and let the crew handle it.
    if (!playerCar && dist < D.exitRange && this.hasLos) {
      this.mode = 'hold';
      this.hold(v);
      if (v.state.speedMs < D.exitSpeed) this.wantsBail = true;
      return;
    }
    if (this.mode === 'hold') {
      // The player drove off or ran: resume the chase (the crew already bailed if it was stopped).
      this.mode = 'route';
    }

    if (dist < D.pursuitRange && this.hasLos) {
      this.mode = 'pursuit';
      this.pursue(v, dt, playerPos, playerCar, dist);
      return;
    }
    this.mode = 'route';
    this.followRoute(v, dt, D.cruiseSpeed);
  }

  // --- Behaviours --------------------------------------------------------------------

  private hold(v: IVehicle): void {
    const c = v.controls;
    c.throttle = 0;
    c.steer = 0;
    c.brake = v.state.speedMs > 0.5 ? 1 : 0;
    c.handbrake = true;
  }

  private pursue(v: IVehicle, dt: number, playerPos: V3, playerCar: IVehicle | null, dist: number): void {
    const lead = playerCar ? D.ramLead : 0;
    const vel = playerCar ? playerCar.velocity : null;
    aim.set(playerPos.x + (vel ? vel.x * lead : 0), playerPos.y, playerPos.z + (vel ? vel.z * lead : 0));
    const ramming = this.ctx.tier.ramming && playerCar !== null;
    let target = D.cruiseSpeed * 1.4;
    if (!ramming) {
      // Keep a gap rather than shove the car; brake down as the gap closes.
      const gap = dist - D.followGap;
      const playerSpeed = playerCar ? playerCar.state.speedMs : 0;
      target = gap < 0 ? 0 : Math.min(target, playerSpeed + Math.sqrt(2 * 6 * gap));
    }
    this.steerAndDrive(v, dt, aim, target, ramming);
  }

  private followRoute(v: IVehicle, dt: number, cruise: number): void {
    this.repathTimer -= dt;
    if (this.repathTimer <= 0) {
      this.repathTimer = D.repathInterval;
      this.replan(v);
    }
    if (this.pathIndex >= this.path.length) {
      // Route exhausted (at the player's node, or no road at all): head straight at them until
      // the next plan; a departing unit just keeps rolling forward instead.
      if (this.mode === 'leave') {
        aim.set(v.position.x + Math.sin(v.heading) * 30, v.position.y, v.position.z + Math.cos(v.heading) * 30);
      } else {
        const p = this.ctx.playerPosition();
        aim.set(p.x, p.y, p.z);
      }
      this.steerAndDrive(v, dt, aim, cruise * 0.5, false);
      return;
    }
    this.laneTarget(v, this.waypoint);
    const wx = this.waypoint.x - v.position.x;
    const wz = this.waypoint.z - v.position.z;
    if (wx * wx + wz * wz < D.nodeReach * D.nodeReach) {
      this.pathIndex++;
      if (this.pathIndex < this.path.length) this.laneTarget(v, this.waypoint);
    }
    this.steerAndDrive(v, dt, this.waypoint, cruise, false);
  }

  private replan(v: IVehicle): void {
    const from = this.graph.nearestNode(v.position);
    const goal = this.mode === 'leave' ? this.awayNode(v) : this.graph.nearestNode(this.ctx.playerPosition());
    const path = this.graph.astar(from.id, goal.id);
    this.path = path ?? [];
    // Skip the start node when the car is already past it, so it does not turn back.
    this.pathIndex = this.path.length > 1 ? 1 : 0;
  }

  /** Farthest neighbour-of-neighbour from the player: a plausible "drive off" destination. */
  private awayNode(v: IVehicle): { id: number } {
    const nodes = this.graph.nodes;
    const p = this.ctx.playerPosition();
    const start = this.graph.nearestNode(v.position);
    let best = start;
    let bestD = -1;
    for (const eid of start.edges) {
      const e = this.graph.edges[eid];
      const nid = e.a === start.id ? e.b : e.a;
      const n = nodes[nid];
      const d = (n.position.x - p.x) ** 2 + (n.position.z - p.z) ** 2;
      if (d > bestD) {
        bestD = d;
        best = n;
      }
    }
    return best;
  }

  /** Node position shifted to the right-hand lane relative to the travel direction. */
  private laneTarget(v: IVehicle, out: Vector3): void {
    const node = this.graph.nodes[this.path[this.pathIndex]].position;
    const prev = this.pathIndex > 0 ? this.graph.nodes[this.path[this.pathIndex - 1]].position : v.position;
    let dx = node.x - prev.x;
    let dz = node.z - prev.z;
    const len = Math.hypot(dx, dz);
    if (len < 1e-3) {
      out.copy(node);
      return;
    }
    dx /= len;
    dz /= len;
    // Right of travel: (dz, -dx) under the +X east / +Z south convention.
    out.set(node.x + dz * D.laneOffset, node.y, node.z - dx * D.laneOffset);
  }

  private steerAndDrive(v: IVehicle, dt: number, target: V3, desiredSpeed: number, flatOut: boolean): void {
    const c = v.controls;
    const speed = v.state.speedMs;
    const desired = Math.atan2(target.x - v.position.x, target.z - v.position.z);
    const err = wrapAngle(desired - v.heading);

    // Stuck against something: back up while counter-steering, then try again.
    if (this.reverseTime > 0) {
      this.reverseTime -= dt;
      c.throttle = 0;
      c.brake = 1;
      c.steer = err > 0 ? 1 : -1;
      return;
    }
    if (speed < D.stuckSpeed && desiredSpeed > 2) {
      this.stuckTime += dt;
      if (this.stuckTime > D.stuckAfter) {
        this.stuckTime = 0;
        this.reverseTime = D.reverseSeconds;
        this.repathTimer = 0;
      }
    } else {
      this.stuckTime = 0;
    }

    c.steer = clamp(-err * D.steerGain, -1, 1);
    let allowed = desiredSpeed;
    if (!flatOut && Math.abs(err) > D.turnSlowAngle) allowed = Math.min(allowed, D.turnSpeed);
    const dv = allowed - speed;
    if (flatOut || dv > 0) {
      c.throttle = flatOut ? 1 : Math.min(1, dv * D.throttleGain);
      c.brake = 0;
    } else {
      c.throttle = 0;
      c.brake = Math.min(1, -dv * D.brakeGain);
    }
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
