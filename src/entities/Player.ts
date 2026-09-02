import RAPIER, { type Collider, type KinematicCharacterController, type RigidBody } from '@dimforge/rapier3d-compat';
import { Group, Quaternion, Vector3 } from 'three';
import { Config } from '../core/Config';
import {
  allocEntityId,
  type CameraTarget,
  type DamageInfo,
  type Entity,
  type IDamageable,
  type IEnterable,
  type V3,
} from '../core/entities';
import type { Game } from '../core/Game';
import { animatePed, buildPed, PED_PALETTES, type PedPose, type PedRig } from '../gfx/PedBuilder';
import { Groups, interactionGroups } from '../physics/Physics';

export interface PlayerMoveInput {
  /** Camera-relative axes: x right, y forward, each -1..1. */
  x: number;
  y: number;
  cameraYaw: number;
  sprint: boolean;
  walk: boolean;
  jump: boolean;
  aiming: boolean;
}

const P = Config.player;
const CAPSULE_HALF_HEIGHT = (P.capsuleHeight - 2 * P.capsuleRadius) / 2;
const CAPSULE_CENTER_Y = P.capsuleHeight / 2;
const UP = new Vector3(0, 1, 0);

/**
 * The protagonist: kinematic character controller on a capsule, procedural rig,
 * health/armor with regen, death → respawn, mount/dismount for vehicles.
 * All simulation happens in `fixedUpdate`; `update` only interpolates visuals.
 */
export class Player implements Entity, IDamageable {
  readonly id = allocEntityId();
  readonly kind = 'player' as const;
  readonly object3d: Group;
  /** Feet position (interpolated for rendering). */
  readonly position = new Vector3();
  isAlive = true;

  readonly body: RigidBody;
  readonly collider: Collider;
  readonly controller: KinematicCharacterController;
  readonly rig: PedRig;

  /** Horizontal velocity (m/s), y is the vertical velocity. */
  readonly velocity = new Vector3();
  heading = 0;
  grounded = false;
  aiming = false;
  health = P.maxHealth;
  armor = P.maxArmor;
  mounted: { vehicle: IEnterable & Entity; seat: number } | null = null;
  deathTimer = 0;
  /** Seconds since last damage (drives regen). */
  sinceDamage = Infinity;
  moveSpeed01 = 0;
  /** Per-character tuning (characters system): 1 = baseline. */
  readonly stats = { sprintMultiplier: 1, recoilMultiplier: 1, damageMultiplier: 1 };

  readonly cameraTarget: CameraTarget;

  private readonly prevPos = new Vector3();
  private readonly currPos = new Vector3();
  private readonly desired = new Vector3();
  private readonly tmp = new Vector3();
  private readonly wantVel = new Vector3();
  private readonly pose: PedPose = { phase: 0, speed01: 0, aiming: false, handsUp: false, fallen: 0 };
  private jumpHeld = false;
  private readonly quat = new Quaternion();

  constructor(private readonly game: Game, spawn: V3, heading: number) {
    this.heading = heading;
    this.rig = buildPed(PED_PALETTES.lucia);
    this.object3d = this.rig.root;
    this.object3d.name = 'player';
    game.actorsRoot.add(this.object3d);

    const physics = game.physics;
    this.body = physics.createKinematicBody({ x: spawn.x, y: spawn.y + CAPSULE_CENTER_Y, z: spawn.z });
    const desc = RAPIER.ColliderDesc.capsule(CAPSULE_HALF_HEIGHT, P.capsuleRadius).setMass(P.mass);
    this.collider = physics.addCollider(desc, this.body, this, {
      groups: Groups.PLAYER,
      filter: Groups.STATIC | Groups.VEHICLE | Groups.PED | Groups.PICKUP | Groups.TRIGGER,
      events: true,
      friction: 0,
    });
    this.controller = physics.world.createCharacterController(Config.physics.characterOffset);
    this.controller.setUp({ x: 0, y: 1, z: 0 });
    this.controller.enableAutostep(P.autostepHeight, P.autostepMinWidth, true);
    this.controller.enableSnapToGround(P.snapToGround);
    this.controller.setApplyImpulsesToDynamicBodies(true);
    this.controller.setCharacterMass(P.mass);
    this.controller.setMaxSlopeClimbAngle((50 * Math.PI) / 180);
    this.controller.setMinSlopeSlideAngle((55 * Math.PI) / 180);
    this.controller.setSlideEnabled(true);

    this.currPos.set(spawn.x, spawn.y, spawn.z);
    this.prevPos.copy(this.currPos);
    this.position.copy(this.currPos);
    this.cameraTarget = { position: this.position, quaternion: this.quat, velocity: this.velocity, kind: 'onFoot' };
    this.syncVisual(1);
  }

  // --- Simulation ------------------------------------------------------------

  fixedUpdate(dt: number, input: PlayerMoveInput): void {
    this.prevPos.copy(this.currPos);
    if (this.mounted) return;
    if (!this.isAlive) {
      // Real time: the WASTED card runs the clock at 0.25x and must line up with the respawn.
      const scale = this.game.time.scale;
      this.deathTimer -= scale > 0 ? dt / scale : dt;
      this.velocity.x = 0;
      this.velocity.z = 0;
      this.applyGravityOnly(dt);
      return;
    }
    this.sinceDamage += dt;
    if (this.sinceDamage >= P.regenDelay && this.health < P.regenTarget) {
      this.health = Math.min(P.regenTarget, this.health + P.regenRate * dt);
    }

    this.aiming = input.aiming;
    const maxSpeed = input.walk ? P.walkSpeed : input.sprint ? P.sprintSpeed * this.stats.sprintMultiplier : P.runSpeed;
    const len = Math.hypot(input.x, input.y);
    const ix = len > 1 ? input.x / len : input.x;
    const iy = len > 1 ? input.y / len : input.y;
    const sinY = Math.sin(input.cameraYaw);
    const cosY = Math.cos(input.cameraYaw);
    // Camera yaw 0 looks toward -Z; "forward" is the camera's horizontal forward.
    const fwdX = -sinY;
    const fwdZ = -cosY;
    const rightX = cosY;
    const rightZ = -sinY;
    this.wantVel.set((fwdX * iy + rightX * ix) * maxSpeed, 0, (fwdZ * iy + rightZ * ix) * maxSpeed);

    const accel = len > 0.01 ? P.acceleration : P.deceleration;
    this.tmp.set(this.wantVel.x - this.velocity.x, 0, this.wantVel.z - this.velocity.z);
    const dvLen = this.tmp.length();
    const maxDv = accel * dt;
    if (dvLen > maxDv) this.tmp.multiplyScalar(maxDv / dvLen);
    this.velocity.x += this.tmp.x;
    this.velocity.z += this.tmp.z;

    if (this.grounded) {
      if (input.jump && !this.jumpHeld) {
        this.velocity.y = P.jumpSpeed;
        this.grounded = false;
      } else if (this.velocity.y < 0) {
        this.velocity.y = -1; // small downward bias keeps snap-to-ground engaged
      }
    } else {
      this.velocity.y = Math.max(P.terminalFall, this.velocity.y + P.gravity * dt);
    }
    this.jumpHeld = input.jump;

    this.moveBody(dt);

    // Facing: toward movement (or camera yaw while aiming), limited turn rate.
    let targetHeading = this.heading;
    // Camera yaw 0 looks toward -Z while rig heading 0 faces +Z, hence the π offset.
    if (this.aiming) targetHeading = input.cameraYaw + Math.PI;
    else if (len > 0.01) targetHeading = Math.atan2(this.wantVel.x, this.wantVel.z);
    this.heading = turnToward(this.heading, targetHeading, P.turnRateRadPerSec * dt);

    const horizontalSpeed = Math.hypot(this.velocity.x, this.velocity.z);
    this.moveSpeed01 = Math.min(1, horizontalSpeed / P.sprintSpeed);
    this.pose.phase += dt * (4 + 6 * this.moveSpeed01) * (horizontalSpeed > 0.2 ? 1 : 0);
  }

  private applyGravityOnly(dt: number): void {
    if (!this.grounded) this.velocity.y = Math.max(P.terminalFall, this.velocity.y + P.gravity * dt);
    else this.velocity.y = -1;
    this.moveBody(dt);
  }

  private moveBody(dt: number): void {
    this.desired.set(this.velocity.x * dt, this.velocity.y * dt, this.velocity.z * dt);
    this.controller.computeColliderMovement(
      this.collider,
      this.desired,
      RAPIER.QueryFilterFlags.EXCLUDE_SENSORS,
      interactionGroups(Groups.PLAYER, Groups.STATIC | Groups.VEHICLE | Groups.PED),
    );
    const move = this.controller.computedMovement();
    this.grounded = this.controller.computedGrounded();
    const t = this.body.translation();
    const nx = t.x + move.x;
    const ny = t.y + move.y;
    const nz = t.z + move.z;
    this.body.setNextKinematicTranslation({ x: nx, y: ny, z: nz });
    this.currPos.set(nx, ny - CAPSULE_CENTER_Y, nz);
    if (this.grounded && this.velocity.y < 0) this.velocity.y = -1;
    // Hitting a ceiling while rising: kill vertical velocity so we do not "stick".
    if (this.velocity.y > 0 && move.y < this.desired.y * 0.5) this.velocity.y = 0;
  }

  // --- Visuals ---------------------------------------------------------------

  update(_dt: number, alpha: number): void {
    this.syncVisual(alpha);
  }

  private syncVisual(alpha: number): void {
    if (this.mounted) {
      this.position.copy(this.mounted.vehicle.position);
      return;
    }
    this.position.lerpVectors(this.prevPos, this.currPos, alpha);
    this.object3d.position.copy(this.position);
    this.object3d.rotation.y = this.heading;
    this.quat.setFromAxisAngle(UP, this.heading);
    this.pose.speed01 = this.isAlive ? this.moveSpeed01 : 0;
    this.pose.aiming = this.aiming && this.isAlive;
    this.pose.fallen = this.isAlive ? 0 : Math.min(1, (P.deathDuration - this.deathTimer) / 0.45);
    animatePed(this.rig, this.pose);
  }

  // --- Damage / death --------------------------------------------------------

  applyDamage(info: DamageInfo): void {
    if (!this.isAlive || (this.mounted !== null && info.type === 'vehicle')) return;
    let amount = info.amount;
    if (this.armor > 0) {
      const absorbed = Math.min(this.armor, amount * P.armorAbsorb);
      this.armor -= absorbed;
      amount -= absorbed;
    }
    this.health -= amount;
    this.sinceDamage = 0;
    // Hurt/death/jump cues are played by the audio system (eventCues / footstepDriver) from these events.
    this.game.events.emit('player:damaged', { amount: info.amount, source: info.source ?? null });
    if (this.health <= 0) this.die();
  }

  private die(): void {
    this.health = 0;
    this.isAlive = false;
    this.deathTimer = P.deathDuration;
    this.velocity.set(0, this.velocity.y, 0);
    this.game.events.emit('player:died', { position: { x: this.position.x, y: this.position.y, z: this.position.z } });
  }

  /** Called by the system when the death timer expires. */
  respawn(position: V3, heading: number, reason: 'wasted' | 'busted'): void {
    this.health = P.maxHealth;
    this.armor = 0;
    this.isAlive = true;
    this.deathTimer = 0;
    this.sinceDamage = Infinity;
    this.teleport(position, heading);
    this.game.events.emit('player:respawned', { position: { x: position.x, y: position.y, z: position.z }, reason });
  }

  // --- Placement -------------------------------------------------------------

  teleport(position: V3, heading?: number): void {
    const center = { x: position.x, y: position.y + CAPSULE_CENTER_Y, z: position.z };
    this.body.setTranslation(center, true);
    this.body.setNextKinematicTranslation(center);
    // The character controller reads the collider's pose, which otherwise lags one step behind.
    this.game.physics.world.propagateModifiedBodyPositionsToColliders();
    this.currPos.set(position.x, position.y, position.z);
    this.prevPos.copy(this.currPos);
    this.velocity.set(0, 0, 0);
    this.grounded = false;
    if (heading !== undefined) this.heading = heading;
    this.syncVisual(1);
  }

  mount(vehicle: IEnterable & Entity, seat: number): void {
    if (this.mounted) return;
    this.mounted = { vehicle, seat };
    this.collider.setEnabled(false);
    this.object3d.visible = false;
    this.velocity.set(0, 0, 0);
    this.aiming = false;
    this.game.events.emit('player:mounted', { vehicle, seat });
  }

  dismount(position: V3): void {
    if (!this.mounted) return;
    this.mounted = null;
    this.collider.setEnabled(true);
    this.object3d.visible = true;
    this.teleport(position);
    this.game.events.emit('player:dismounted', { position: { x: position.x, y: position.y, z: position.z } });
  }

  dispose(): void {
    this.game.actorsRoot.remove(this.object3d);
    this.game.physics.world.removeCharacterController(this.controller);
    this.game.physics.removeBody(this.body);
  }
}

function turnToward(current: number, target: number, maxDelta: number): number {
  let diff = target - current;
  while (diff > Math.PI) diff -= Math.PI * 2;
  while (diff < -Math.PI) diff += Math.PI * 2;
  if (Math.abs(diff) <= maxDelta) return target;
  return current + Math.sign(diff) * maxDelta;
}
