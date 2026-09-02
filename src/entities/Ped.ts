import RAPIER, { type Collider, type KinematicCharacterController, type RigidBody } from '@dimforge/rapier3d-compat';
import { Quaternion, Vector3, type Object3D } from 'three';
import {
  allocEntityId,
  type DamageInfo,
  type Entity,
  type IPed,
  type IVehicle,
  type PedBrain,
  type PedEvent,
  type PedState,
  type V3,
} from '../core/entities';
import type { Game } from '../core/Game';
import type { WeaponId } from '../core/peers';
import type { Random } from '../core/Random';
import { PED_TYPES, type PedTypeDef } from '../data/peds';
import type { PedPalette, PedPose } from '../gfx/PedBuilder';
import { Groups, interactionGroups } from '../physics/Physics';
import { PedConfig } from '../systems/peds/config';
import { damagedEvent } from '../systems/peds/events';
import { stepPedMotion } from '../systems/peds/PedMotion';

const B = PedConfig.body;
const R = PedConfig.ragdoll;
const CAPSULE_HALF_HEIGHT = (B.capsuleHeight - 2 * B.capsuleRadius) / 2;
export const PED_CAPSULE_CENTER_Y = B.capsuleHeight / 2;
const CAPSULE_CENTER_Y = PED_CAPSULE_CENTER_Y;
const ALIVE_FILTER = Groups.STATIC | Groups.PLAYER | Groups.VEHICLE | Groups.PED;
/** Knocked-down peds stop blocking cars so a slow bump does not park the vehicle on them. */
const KNOCKED_FILTER = Groups.STATIC | Groups.PLAYER | Groups.PED;
/** Character-controller filter: peds slide around geometry, cars, the player and each other. */
export const PED_MOVE_GROUPS = interactionGroups(Groups.PED, Groups.STATIC | Groups.PLAYER | Groups.VEHICLE | Groups.PED);
const GETUP_SECONDS = 0.6;
const tmpV = new Vector3();
const tmpV2 = new Vector3();

/** A brain that does nothing; assigned while a ped sits in the pool. */
const NULL_BRAIN: PedBrain = { update: () => {} };

/**
 * Pooled pedestrian: kinematic capsule while alive (moved by the system's shared
 * character controller), dynamic capsule while ragdolling, instanced visuals owned by
 * the renderer. Brains drive it only through the IPed motion primitives.
 */
export class Ped implements IPed {
  readonly id = allocEntityId();
  readonly kind = 'ped' as const;
  object3d: Object3D | null = null;
  /** Feet position (interpolated for rendering). */
  readonly position = new Vector3();
  isAlive = false;

  typeId = 'beachgoer';
  type: PedTypeDef = PED_TYPES.beachgoer;
  palette: PedPalette = PED_TYPES.beachgoer.palettes[0];
  private currentState: PedState = 'wander';
  brain: PedBrain = NULL_BRAIN;
  heading = 0;
  health: number = B.maxHealth;
  readonly velocity = new Vector3();
  weaponId: WeaponId | null = null;
  handsUp = false;
  isCop = false;
  vehicle: IVehicle | null = null;
  seat = 0;
  /** Used by the damage system for headshot classification. */
  readonly headHeight = B.headHeight;

  body: RigidBody | null = null;
  collider: Collider | null = null;
  /** Slot in the instanced renderer; -1 while pooled. */
  instanceIndex = -1;
  /** True between spawn and despawn. */
  active = false;
  grounded = false;
  /** Seconds since last damage. */
  sinceDamage = Infinity;
  /** Seconds in the current state (reset by `setState`). */
  stateTime = 0;
  ragdolled = false;
  frozen = false;
  ragdollTime = 0;
  /** Set when the ragdoll has run its course; the system despawns the ped. */
  wantsDespawn = false;
  knockdownTimer = 0;
  speed01 = 0;
  readonly pose: PedPose = { phase: 0, speed01: 0, aiming: false, handsUp: false, fallen: 0 };
  /** Rigid-body orientation while ragdolled (renderer reads it). */
  readonly bodyQuat = new Quaternion();

  // Motion scratch state, written by `stepPedMotion` (systems/peds/PedMotion.ts); not for brains.
  moving = false;
  facing = false;
  moveSpeed = 0;
  vy = 0;
  readonly moveTarget = new Vector3();
  readonly faceTarget = new Vector3();
  readonly wantVel = new Vector3();
  readonly desired = new Vector3();
  readonly prevPos = new Vector3();
  readonly currPos = new Vector3();
  private pendingHitImpulse = 0;

  constructor(
    private readonly game: Game,
    private readonly rng: Random,
  ) {}

  // --- Lifecycle (called by PedSystem) --------------------------------------------

  spawn(type: PedTypeDef, palette: PedPalette, position: V3, heading: number, brain: PedBrain): void {
    this.ensureBody();
    const body = this.body as RigidBody;
    const collider = this.collider as Collider;
    this.type = type;
    this.typeId = type.id;
    this.palette = palette;
    this.isCop = type.isCop;
    this.weaponId = type.defaultWeapon;
    this.health = B.maxHealth;
    this.isAlive = true;
    this.active = true;
    this.handsUp = false;
    this.vehicle = null;
    this.seat = 0;
    this.moving = false;
    this.facing = false;
    this.ragdolled = false;
    this.frozen = false;
    this.ragdollTime = 0;
    this.wantsDespawn = false;
    this.knockdownTimer = 0;
    this.pendingHitImpulse = 0;
    this.sinceDamage = Infinity;
    this.speed01 = 0;
    this.velocity.set(0, 0, 0);
    this.pose.phase = this.rng.range(0, Math.PI * 2);
    this.pose.speed01 = 0;
    this.pose.aiming = false;
    this.pose.handsUp = false;
    this.pose.fallen = 0;
    this.bodyQuat.identity();
    body.setEnabled(true);
    body.setBodyType(RAPIER.RigidBodyType.KinematicPositionBased, true);
    body.setRotation({ x: 0, y: 0, z: 0, w: 1 }, false);
    collider.setEnabled(true);
    collider.setCollisionGroups(interactionGroups(Groups.PED, ALIVE_FILTER));
    this.brain = brain;
    this.currentState = 'wander';
    this.stateTime = 0;
    this.teleport(position, heading);
  }

  /** Returns the ped to the pool: body disabled, brain detached. */
  despawn(): void {
    this.active = false;
    this.isAlive = false;
    this.brain = NULL_BRAIN;
    if (this.vehicle) {
      this.vehicle.exit(this);
      this.vehicle = null;
    }
    if (this.body) {
      this.body.setBodyType(RAPIER.RigidBodyType.KinematicPositionBased, false);
      this.body.setEnabled(false);
    }
    this.collider?.setEnabled(false);
  }

  private ensureBody(): void {
    if (this.body) return;
    const physics = this.game.physics;
    this.body = physics.createKinematicBody({ x: 0, y: -50, z: 0 });
    const desc = RAPIER.ColliderDesc.capsule(CAPSULE_HALF_HEIGHT, B.capsuleRadius).setMass(B.mass).setFriction(0.6);
    this.collider = physics.addCollider(desc, this.body, this, { groups: Groups.PED, filter: ALIVE_FILTER, events: true });
    // Traffic cars are kinematic too; without this flag a kinematic car never reports hitting a ped.
    this.collider.setActiveCollisionTypes(RAPIER.ActiveCollisionTypes.DEFAULT | RAPIER.ActiveCollisionTypes.KINEMATIC_KINEMATIC);
  }

  dispose(): void {
    if (this.body) this.game.physics.removeBody(this.body);
    this.body = null;
    this.collider = null;
  }

  // --- IPed primitives --------------------------------------------------------------

  setBrain(brain: PedBrain): void {
    this.brain = brain;
  }

  get state(): PedState {
    return this.currentState;
  }

  /** Assigning a new state resets `stateTime`; same-state writes are no-ops. */
  set state(state: PedState) {
    if (this.currentState === state) return;
    this.currentState = state;
    this.stateTime = 0;
  }

  setState(state: PedState): void {
    this.state = state;
  }

  moveToward(target: V3, speedMs: number): void {
    this.moveTarget.set(target.x, target.y, target.z);
    this.moveSpeed = speedMs;
    this.moving = true;
  }

  stop(): void {
    this.moving = false;
  }

  face(target: V3): void {
    this.faceTarget.set(target.x, target.y, target.z);
    this.facing = true;
  }

  /** Forward unit vector for the current heading (rig heading 0 faces +Z). */
  forward(out: Vector3): Vector3 {
    return out.set(Math.sin(this.heading), 0, Math.cos(this.heading));
  }

  notify(event: PedEvent): void {
    this.brain.onEvent?.(this, event);
  }

  kill(source?: Entity | 'player'): void {
    this.applyDamage({ amount: 1e6, type: 'bullet', source });
  }

  enterVehicle(vehicle: IVehicle, seat = 0): boolean {
    if (!this.isAlive || this.vehicle || this.ragdolled) return false;
    const taken = vehicle.enter(this, seat);
    if (taken < 0) return false;
    this.vehicle = vehicle;
    this.seat = taken;
    this.moving = false;
    this.facing = false;
    this.handsUp = false;
    this.velocity.set(0, 0, 0);
    this.collider?.setEnabled(false);
    this.setState('inVehicle');
    return true;
  }

  exitVehicle(): void {
    const v = this.vehicle;
    if (!v) return;
    v.exit(this);
    v.seatDoorPosition(this.seat, tmpV);
    this.vehicle = null;
    this.collider?.setEnabled(true);
    this.teleport(tmpV, v.heading);
    if (this.isAlive) this.setState('wander');
  }

  teleport(position: V3, heading?: number): void {
    const body = this.body;
    if (!body) return;
    const center = { x: position.x, y: position.y + CAPSULE_CENTER_Y, z: position.z };
    body.setTranslation(center, true);
    body.setNextKinematicTranslation(center);
    this.game.physics.world.propagateModifiedBodyPositionsToColliders();
    this.currPos.set(position.x, position.y, position.z);
    this.prevPos.copy(this.currPos);
    this.position.copy(this.currPos);
    this.velocity.set(0, 0, 0);
    this.vy = 0;
    this.grounded = false;
    if (heading !== undefined) this.heading = heading;
  }

  // --- Damage -----------------------------------------------------------------------

  applyDamage(info: DamageInfo): void {
    if (!this.isAlive || !this.active) return;
    this.health -= info.amount;
    this.sinceDamage = 0;
    this.game.events.emit('ped:attacked', { ped: this, source: info.source ?? null });
    this.game.audio.play('ped.hit', { position: this.position });
    if (info.point) tmpV.set(info.point.x, info.point.y, info.point.z);
    else tmpV.set(this.position.x, this.position.y + B.chestHeight, this.position.z);
    this.game.particles?.emit('blood', tmpV, { direction: info.direction });
    if (this.health <= 0) {
      this.die(info);
      return;
    }
    this.brain.onEvent?.(this, damagedEvent(info));
  }

  private die(info: DamageInfo): void {
    this.health = 0;
    this.isAlive = false;
    this.handsUp = false;
    this.moving = false;
    this.facing = false;
    if (this.vehicle) this.exitVehicle();
    this.setState('dead');
    const dir = info.direction;
    if (dir && (dir.x !== 0 || dir.y !== 0 || dir.z !== 0)) tmpV.set(dir.x, 0, dir.z).normalize();
    else this.forward(tmpV);
    tmpV.multiplyScalar(R.deathImpulse);
    tmpV.y += R.deathUpImpulse;
    // A ped already tumbling from a car hit keeps that momentum instead of a second kick.
    if (!this.ragdolled) this.ragdoll(tmpV);
    const source = info.source ?? null;
    this.game.events.emit('ped:died', { ped: this, source });
    this.game.audio.play('ped.death', { position: this.position });
    // The damage system attributes kills when loaded; only fall back when it is absent.
    if (!this.game.damage && source === 'player') {
      this.game.events.emit('crime:committed', {
        type: this.isCop ? 'kill_cop' : 'kill_ped',
        position: { x: this.position.x, y: this.position.y, z: this.position.z },
        actor: 'player',
        loud: false,
      });
    }
    const G = PedConfig.gang;
    if (this.type.id === 'gang' && this.rng.chance(G.dropChance)) this.dropCash(this.rng.int(G.dropMin, G.dropMax));
  }

  /** Spawns a cash pickup just in front of the ped (robbery, gang loot). */
  dropCash(amount: number): void {
    this.forward(tmpV2).multiplyScalar(PedConfig.cower.dropForward);
    tmpV2.add(this.position);
    tmpV2.y += 0.3;
    this.game.pickups?.spawn('cash', tmpV2, amount);
    this.game.events.emit('ped:droppedCash', { ped: this, amount });
  }

  ragdoll(impulse: V3): void {
    const body = this.body;
    const collider = this.collider;
    if (!body || !collider || !this.active || this.state === 'inVehicle') return;
    if (!this.ragdolled) {
      this.ragdolled = true;
      this.frozen = false;
      this.ragdollTime = 0;
      this.moving = false;
      this.facing = false;
      this.handsUp = false;
      this.velocity.set(0, 0, 0);
      this.pose.fallen = 0;
      collider.setCollisionGroups(interactionGroups(Groups.PED, ALIVE_FILTER));
      body.setBodyType(RAPIER.RigidBodyType.Dynamic, true);
      body.setEnabledRotations(true, true, true, true);
      body.setLinearDamping(R.linearDamping);
      body.setAngularDamping(R.angularDamping);
      if (this.isAlive) this.setState('knockedDown');
      this.game.audio.play('ped.body.fall', { position: this.position });
    }
    body.applyImpulse(impulse, true);
    // Tumble around the axis perpendicular to the push so the capsule topples instead of sliding.
    const mag = Math.hypot(impulse.x, impulse.z);
    if (mag > 1e-3) {
      body.applyTorqueImpulse({ x: (impulse.z / mag) * R.deathTorque, y: 0, z: (-impulse.x / mag) * R.deathTorque }, true);
    } else {
      body.applyTorqueImpulse({ x: -R.deathTorque, y: 0, z: 0 }, true);
    }
  }

  /** Low-speed vehicle bump: fall over, get up after a while, then the brain reacts. */
  knockDown(source: Entity | null, impulse: number): void {
    if (!this.isAlive || this.ragdolled || this.state === 'inVehicle') return;
    this.setState('knockedDown');
    this.knockdownTimer = PedConfig.knockdown.duration;
    this.pendingHitImpulse = impulse;
    this.moving = false;
    this.facing = false;
    this.handsUp = false;
    this.velocity.set(0, 0, 0);
    this.collider?.setCollisionGroups(interactionGroups(Groups.PED, KNOCKED_FILTER));
    this.game.audio.play('ped.knockdown', { position: this.position });
    this.game.events.emit('ped:knockedDown', { ped: this, source });
  }

  private standUp(): void {
    const body = this.body as RigidBody;
    const t = body.translation();
    const ground = this.game.physics.groundHeightAt(t.x, t.z, t.y + 1, 6);
    body.setBodyType(RAPIER.RigidBodyType.KinematicPositionBased, true);
    body.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
    body.setLinvel({ x: 0, y: 0, z: 0 }, false);
    body.setAngvel({ x: 0, y: 0, z: 0 }, false);
    this.ragdolled = false;
    this.bodyQuat.identity();
    this.teleport({ x: t.x, y: ground ?? t.y - CAPSULE_CENTER_Y, z: t.z });
    this.knockdownTimer = GETUP_SECONDS;
    this.pose.fallen = 1;
    this.setState('knockedDown');
  }

  // --- Simulation -------------------------------------------------------------------

  fixedUpdate(dt: number, controller: KinematicCharacterController): void {
    const body = this.body;
    const collider = this.collider;
    if (!body || !collider || !this.active) return;
    this.prevPos.copy(this.currPos);
    this.stateTime += dt;
    this.sinceDamage += dt;

    if (this.ragdolled) {
      this.updateRagdoll(dt, body);
      return;
    }
    if (this.vehicle) {
      const p = this.vehicle.position;
      this.currPos.set(p.x, p.y, p.z);
      this.prevPos.copy(this.currPos);
      body.setTranslation({ x: p.x, y: p.y + CAPSULE_CENTER_Y, z: p.z }, false);
      this.heading = this.vehicle.heading;
      return;
    }
    if (!this.isAlive) return;
    stepPedMotion(this, dt, controller, body, collider);
  }

  /** Knockdown timer expired: stand back up and let the brain react (called by `stepPedMotion`). */
  finishKnockdown(): void {
    this.collider?.setCollisionGroups(interactionGroups(Groups.PED, ALIVE_FILTER));
    this.game.audio.play('ped.getup', { position: this.position });
    this.setState('wander');
    const impulse = this.pendingHitImpulse;
    this.pendingHitImpulse = 0;
    this.brain.onEvent?.(this, { type: 'hitByVehicle', impulse });
  }

  private updateRagdoll(dt: number, body: RigidBody): void {
    this.ragdollTime += dt;
    const t = body.translation();
    const r = body.rotation();
    this.bodyQuat.set(r.x, r.y, r.z, r.w);
    this.currPos.set(0, -CAPSULE_CENTER_Y, 0).applyQuaternion(this.bodyQuat);
    this.currPos.x += t.x;
    this.currPos.y += t.y;
    this.currPos.z += t.z;
    this.prevPos.copy(this.currPos);
    this.position.copy(this.currPos);
    const lv = body.linvel();
    this.velocity.set(lv.x, 0, lv.z);
    if (!this.frozen && this.ragdollTime >= R.freezeAfter) {
      if (this.isAlive) {
        this.standUp();
        return;
      }
      this.frozen = true;
      body.setBodyType(RAPIER.RigidBodyType.Fixed, true);
      // A frozen corpse must not become a wall for cars and the player; bullets still hit it.
      this.collider?.setCollisionGroups(interactionGroups(Groups.PED, Groups.STATIC));
    }
    if (this.frozen && this.ragdollTime >= R.freezeAfter + R.despawnAfter) this.wantsDespawn = true;
  }

  // --- Visual state (renderer reads `pose`, `position`, `heading`, `bodyQuat`) -------

  updateVisual(dt: number, alpha: number, animate: boolean): void {
    if (!this.ragdolled) this.position.lerpVectors(this.prevPos, this.currPos, alpha);
    const p = this.pose;
    p.speed01 = animate && this.isAlive && this.state !== 'knockedDown' ? this.speed01 : 0;
    p.handsUp = this.handsUp && this.isAlive;
    p.aiming = !p.handsUp && this.isAlive && this.weaponId !== null && this.state === 'fight';
    const fallenTarget = this.state === 'knockedDown' && !this.ragdolled ? 1 : 0;
    const step = dt / 0.35;
    if (p.fallen < fallenTarget) p.fallen = Math.min(fallenTarget, p.fallen + step);
    else if (p.fallen > fallenTarget) p.fallen = Math.max(fallenTarget, p.fallen - step);
  }
}
