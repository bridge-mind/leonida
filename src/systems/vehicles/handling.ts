import type { VehicleControls } from '../../core/entities';
import type { VehicleArchetype } from '../../data/vehicles';
import { VehicleConfig } from './config';

/**
 * Arcade handling model (the sanctioned fallback from SYSTEM_PROMPT §5): the body is
 * one yaw-locked dynamic rigid body and this function rewrites its velocity in the
 * car's own frame each fixed step. Pure so it can be unit-tested without Rapier.
 */
export interface HandlingState {
  /** Forward speed (m/s), negative when reversing. */
  vf: number;
  /** Lateral speed (m/s), positive toward the car's right. */
  vl: number;
  /** Yaw rate (rad/s); positive turns toward +X when facing +Z (a left turn). */
  yawRate: number;
}

const H = VehicleConfig.handling;

export function stepHandling(s: HandlingState, a: VehicleArchetype, c: VehicleControls, dt: number, grounded: boolean): void {
  let vf = s.vf;
  let vl = s.vl;
  const speed = Math.abs(vf);

  if (grounded) {
    if (c.throttle > 0 && vf > -H.brakeToReverseSpeed) {
      const ratio = Math.min(1, speed / a.topSpeed);
      vf += c.throttle * a.engineAccel * (1 - H.topSpeedFalloff * ratio * ratio) * dt;
    }
    if (c.brake > 0) {
      if (vf > H.brakeToReverseSpeed) {
        vf = Math.max(0, vf - c.brake * a.brake * dt);
      } else {
        const ratio = Math.min(1, speed / a.reverseSpeed);
        vf = Math.max(-a.reverseSpeed, vf - c.brake * a.engineAccel * 0.6 * (1 - ratio) * dt);
      }
    }
    if (c.handbrake) vf = approachZero(vf, H.handbrakeDecel * dt);
    vf = approachZero(vf, (H.drag * vf * vf + H.rolling) * dt);
    if (vf > a.topSpeed) vf = a.topSpeed;
    else if (vf < -a.reverseSpeed) vf = -a.reverseSpeed;

    // Tire grip bleeds sideways velocity; the handbrake lets the rear step out.
    const grip = c.handbrake ? a.handbrakeGrip : a.grip;
    vl *= Math.pow(1 - grip, dt * 60);
  } else {
    // Airborne: only aero drag, no steering authority, keep drifting on the current vector.
    vf = approachZero(vf, H.drag * vf * vf * dt);
  }

  const authority = grounded
    ? (Math.min(1, speed / H.minSteerSpeed) / (1 + speed / H.steerFalloffSpeed)) * (c.handbrake ? H.handbrakeSteerBoost : 1)
    : 0;
  const travelSign = vf >= 0 ? 1 : -1;
  // steer > 0 is right; with forward = (sin h, cos h) a positive yaw rate is a left turn.
  const targetYaw = -c.steer * a.maxSteerRate * authority * travelSign;
  s.yawRate += (targetYaw - s.yawRate) * (1 - Math.exp(-H.yawResponse * dt));
  s.vf = vf;
  s.vl = vl;
}

function approachZero(v: number, amount: number): number {
  if (v > 0) return Math.max(0, v - amount);
  if (v < 0) return Math.min(0, v + amount);
  return 0;
}

/** Engine rpm 0..1 shaped into gears so the pitch climbs and drops like a real gearbox. */
export function rpmFromSpeed(speedMs: number, topSpeed: number, throttle: number): number {
  const ratio = Math.min(1, speedMs / topSpeed);
  const gears = VehicleConfig.audio.gears;
  const g = Math.min(gears - 1, Math.floor(ratio * gears));
  const within = ratio * gears - g;
  const idle = VehicleConfig.audio.idleRpm;
  const revving = speedMs < 1 ? throttle * 0.5 : 0;
  return Math.min(1, idle + (1 - idle) * (0.25 + 0.75 * within) * Math.min(1, ratio * 6 + 0.15) + revving);
}

/** Wraps an angle to (-π, π]. */
export function wrapAngle(a: number): number {
  while (a > Math.PI) a -= Math.PI * 2;
  while (a <= -Math.PI) a += Math.PI * 2;
  return a;
}

export function lerpAngle(a: number, b: number, t: number): number {
  return a + wrapAngle(b - a) * t;
}
