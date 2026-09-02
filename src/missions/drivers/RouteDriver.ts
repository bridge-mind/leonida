import { Vector3 } from 'three';
import type { IVehicle, V3, VehicleDriver } from '../../core/entities';

const aim = new Vector3();

function wrap(a: number): number {
  while (a > Math.PI) a -= Math.PI * 2;
  while (a < -Math.PI) a += Math.PI * 2;
  return a;
}

/** Occupies the driver seat without moving: parked cars that count as "owned" (carjack prompt). */
export class IdleDriver implements VehicleDriver {
  update(v: IVehicle): void {
    v.controls.throttle = 0;
    v.controls.brake = 1;
    v.controls.handbrake = true;
  }
}

/**
 * Scripted driver that follows a fixed polyline (a `roadGraph.routeBetween` result) with a
 * lookahead point, then brakes at the end. Simpler than TrafficDriver: no lights, no
 * lane changes, no car-following — mission chases want predictability over etiquette.
 */
export class RouteDriver implements VehicleDriver {
  private seg = 0;
  done = false;

  constructor(
    private readonly points: readonly V3[],
    private readonly speed = 16,
    private readonly lookahead = 9,
  ) {}

  update(v: IVehicle, dt: number): void {
    void dt;
    const pts = this.points;
    const c = v.controls;
    if (pts.length < 2 || this.done) {
      c.throttle = 0;
      c.brake = 1;
      c.handbrake = true;
      return;
    }
    const pos = v.position;
    // Advance past segments whose end we have reached.
    while (this.seg < pts.length - 1 && this.projectT(pts[this.seg], pts[this.seg + 1], pos) >= 1) this.seg++;
    if (this.seg >= pts.length - 1) {
      const end = pts[pts.length - 1];
      if (Math.hypot(end.x - pos.x, end.z - pos.z) < 3 || v.state.speedMs < 0.5) this.done = true;
      c.throttle = 0;
      c.brake = 1;
      return;
    }
    this.pointAhead(pos, aim);
    const desired = Math.atan2(aim.x - pos.x, aim.z - pos.z);
    const err = wrap(desired - v.heading);
    c.steer = Math.max(-1, Math.min(1, -err * 1.6));
    const remaining = this.distanceToEnd(pos);
    const turnSlow = Math.abs(err) > 0.5 ? 7 : this.speed;
    const target = Math.min(turnSlow, remaining < 20 ? Math.max(3, remaining * 0.5) : this.speed);
    const dv = target - v.state.speedMs;
    if (dv > 0) {
      c.throttle = Math.min(1, dv * 0.35);
      c.brake = 0;
    } else {
      c.throttle = 0;
      c.brake = Math.min(1, -dv * 0.3);
    }
    c.handbrake = false;
  }

  private projectT(a: V3, b: V3, p: V3): number {
    const dx = b.x - a.x;
    const dz = b.z - a.z;
    const len2 = dx * dx + dz * dz;
    if (len2 < 1e-9) return 1;
    return ((p.x - a.x) * dx + (p.z - a.z) * dz) / len2;
  }

  private pointAhead(pos: V3, out: Vector3): void {
    const pts = this.points;
    let i = this.seg;
    let from = Math.max(0, this.projectT(pts[i], pts[i + 1], pos));
    let remaining = this.lookahead;
    while (i < pts.length - 1) {
      const a = pts[i];
      const b = pts[i + 1];
      const len = Math.hypot(b.x - a.x, b.z - a.z);
      const left = (1 - from) * len;
      if (remaining <= left) {
        const u = from + remaining / Math.max(1e-6, len);
        out.set(a.x + (b.x - a.x) * u, a.y, a.z + (b.z - a.z) * u);
        return;
      }
      remaining -= left;
      from = 0;
      i++;
    }
    const end = pts[pts.length - 1];
    out.set(end.x, end.y, end.z);
  }

  private distanceToEnd(pos: V3): number {
    const pts = this.points;
    let d = Math.hypot(pts[this.seg + 1].x - pos.x, pts[this.seg + 1].z - pos.z);
    for (let i = this.seg + 1; i < pts.length - 1; i++) d += Math.hypot(pts[i + 1].x - pts[i].x, pts[i + 1].z - pts[i].z);
    return d;
  }
}
