import RAPIER, { type Collider, type KinematicCharacterController, type RigidBody } from '@dimforge/rapier3d-compat';
import { PED_CAPSULE_CENTER_Y, PED_MOVE_GROUPS, type Ped } from '../../entities/Ped';
import { PedConfig } from './config';

const M = PedConfig.motion;

/**
 * One kinematic step for an alive, non-ragdolled ped: resolve the brain's move/face
 * request into a velocity, integrate gravity, and slide the capsule with the shared
 * character controller. Split out of `Ped` to keep the entity file focused on state.
 */
export function stepPedMotion(ped: Ped, dt: number, controller: KinematicCharacterController, body: RigidBody, collider: Collider): void {
  const want = ped.wantVel;
  if (ped.state === 'knockedDown') {
    ped.knockdownTimer -= dt;
    want.set(0, 0, 0);
    if (ped.knockdownTimer <= 0) ped.finishKnockdown();
  } else if (ped.moving) {
    const dx = ped.moveTarget.x - ped.currPos.x;
    const dz = ped.moveTarget.z - ped.currPos.z;
    const dist = Math.hypot(dx, dz);
    if (dist <= M.arriveRadius) {
      ped.moving = false;
      want.set(0, 0, 0);
    } else {
      // Ease in over the last meter so arrivals do not overshoot and oscillate.
      const speed = Math.min(ped.moveSpeed, Math.max(0.6, dist * 3));
      want.set((dx / dist) * speed, 0, (dz / dist) * speed);
    }
  } else {
    want.set(0, 0, 0);
  }

  const vel = ped.velocity;
  const accel = ped.moving ? M.acceleration : M.deceleration;
  const dvx = want.x - vel.x;
  const dvz = want.z - vel.z;
  const dvLen = Math.hypot(dvx, dvz);
  const maxDv = accel * dt;
  if (dvLen > maxDv) {
    vel.x += (dvx / dvLen) * maxDv;
    vel.z += (dvz / dvLen) * maxDv;
  } else {
    vel.x = want.x;
    vel.z = want.z;
  }

  const hspeed = Math.hypot(vel.x, vel.z);
  if (hspeed > 0.3) {
    ped.heading = turnToward(ped.heading, Math.atan2(vel.x, vel.z), M.turnRate * dt);
  } else if (ped.facing) {
    const fx = ped.faceTarget.x - ped.currPos.x;
    const fz = ped.faceTarget.z - ped.currPos.z;
    if (fx * fx + fz * fz > 0.01) ped.heading = turnToward(ped.heading, Math.atan2(fx, fz), M.turnRate * dt);
  }

  if (ped.grounded) ped.vy = -1;
  else ped.vy = Math.max(M.terminalFall, ped.vy + M.gravity * dt);

  ped.desired.set(vel.x * dt, ped.vy * dt, vel.z * dt);
  controller.computeColliderMovement(collider, ped.desired, RAPIER.QueryFilterFlags.EXCLUDE_SENSORS, PED_MOVE_GROUPS);
  const move = controller.computedMovement();
  ped.grounded = controller.computedGrounded();
  const t = body.translation();
  const nx = t.x + move.x;
  const ny = t.y + move.y;
  const nz = t.z + move.z;
  body.setNextKinematicTranslation({ x: nx, y: ny, z: nz });
  ped.currPos.set(nx, ny - PED_CAPSULE_CENTER_Y, nz);

  ped.speed01 = Math.min(1, hspeed / ped.type.runSpeed);
  if (hspeed > 0.2) ped.pose.phase += dt * (4 + 6 * ped.speed01);
}

export function turnToward(current: number, target: number, maxDelta: number): number {
  let diff = target - current;
  while (diff > Math.PI) diff -= Math.PI * 2;
  while (diff < -Math.PI) diff += Math.PI * 2;
  if (Math.abs(diff) <= maxDelta) return target;
  return current + Math.sign(diff) * maxDelta;
}
