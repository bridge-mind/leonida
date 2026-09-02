import RAPIER, { type Collider, type RigidBody } from '@dimforge/rapier3d-compat';
import { Group, Quaternion, Vector3 } from 'three';
import {
  allocEntityId,
  type CameraTarget,
  type DamageInfo,
  type Entity,
  type IVehicle,
  type V3,
  type VehicleControls,
  type VehicleDriver,
  type VehicleStateInfo,
} from '../core/entities';
import type { Game } from '../core/Game';
import type { VehicleArchetypeId } from '../core/peers';
import { VEHICLES, type VehicleArchetype } from '../data/vehicles';
import { carLayout, headingQuaternion, seatDoorLocal, type CarLayout } from '../gfx/CarBuilder';
import { Groups, type RaycastHit } from '../physics/Physics';
import { VehicleConfig } from '../systems/vehicles/config';
import { explodeVehicle } from '../systems/vehicles/VehicleFx';
import { lerpAngle, rpmFromSpeed, stepHandling, wrapAngle, type HandlingState } from '../systems/vehicles/handling';

declare module '../core/entities' {
  interface VehicleStateInfo {
    wrecked: boolean;
    lightsOn: boolean;
    sirenOn: boolean;
    /** Sideways speed (m/s); large values mean the car is sliding. */
    lateralSpeed: number;
    /** Signed forward speed (m/s); negative when reversing. */
    forwardSpeed: number;
    grounded: boolean;
  }
}

const C = VehicleConfig;
const DOWN = { x: 0, y: -1, z: 0 };

/**
 * One class for every car: parked (no driver), traffic (`TrafficDriver`), the
 * player (`playerDriven`, controls written by VehicleSystem) and later police.
 * Physics: a yaw-locked dynamic cuboid whose velocity is rewritten by the arcade
 * handling model each fixed step; Rapier resolves collisions and gravity.
 * Rendering is done by the shared instanced VehicleRenderer through `slot`.
 */
export class Vehicle implements IVehicle {
  private entityId = 0;
  readonly kind = 'vehicle' as const;
  /** Empty group tracking the car pose so other systems can parent effects to it. */
  readonly object3d = new Group();
  /** Ground point under the chassis center (interpolated for rendering). */
  readonly position = new Vector3();
  readonly quaternion = new Quaternion();
  readonly velocity = new Vector3();
  isAlive = false;

  archetype: VehicleArchetype = VEHICLES.sedan;
  layout: CarLayout = carLayout('sedan');
  controls: VehicleControls = { throttle: 0, brake: 0, steer: 0, handbrake: false };
  readonly state: VehicleStateInfo = {
    speedMs: 0,
    rpm01: 0,
    health: 1000,
    onFire: false,
    exploded: false,
    occupants: [],
    wrecked: false,
    lightsOn: false,
    sirenOn: false,
    lateralSpeed: 0,
    forwardSpeed: 0,
    grounded: true,
  };
  driver: VehicleDriver | null = null;
  color = 0xffffff;
  readonly cameraTarget: CameraTarget;

  body!: RigidBody;
  collider!: Collider;
  /** Instance slot in the VehicleRenderer, -1 while pooled. */
  slot = -1;
  playerDriven = false;
  /** Headlights: null follows the night level, true/false is the L toggle. */
  lightsOverride: boolean | null = null;
  /** Ped type of the virtual traffic driver (spawned as a real ped on carjack/panic). */
  driverTypeId: string | null = null;
  wheelSpin = 0;
  steerVisual = 0;
  /** Pitch matched to the ground under the axles (visual only, player car). */
  visualPitch = 0;
  readonly hs: HandlingState = { vf: 0, vl: 0, yawRate: 0 };
  /** Velocity before the current physics step; contact handlers diff against it. */
  readonly prevLinvel = new Vector3();
  lastDamageByPlayer = false;
  fireTimer = 0;
  smokeFxTimer = 0;
  fireFxTimer = 0;
  wreckTimer = 0;

  private readonly seats: (Entity | null)[] = [];
  private readonly currPos = new Vector3();
  private readonly prevPos = new Vector3();
  private currHeading = 0;
  private prevHeading = 0;
  private readonly tmp = new Vector3();
  private readonly hit: RaycastHit = { entity: null, point: new Vector3(), normal: new Vector3(), toi: 0, collider: null as unknown as Collider };

  constructor(readonly game: Game) {
    this.object3d.name = 'vehicle';
    this.object3d.visible = false;
    game.worldRoot.add(this.object3d);
    this.cameraTarget = { position: this.position, quaternion: this.quaternion, velocity: this.velocity, kind: 'vehicle', size: 4.5 };
  }

  // --- Identity / geometry -----------------------------------------------------

  get id(): number {
    return this.entityId;
  }

  get archetypeId(): VehicleArchetypeId {
    return this.archetype.id;
  }

  get isPolice(): boolean {
    return this.archetype.id === 'police' || this.archetype.id === 'swat';
  }

  get length(): number {
    return this.archetype.dims.length;
  }

  get width(): number {
    return this.archetype.dims.width;
  }

  get heading(): number {
    return this.currHeading;
  }

  set heading(value: number) {
    this.teleport(this.currPos, value);
  }

  get driverEntity(): Entity | null {
    return this.seats[0] ?? null;
  }

  /** True while a driver object or the player is steering. */
  get hasController(): boolean {
    return this.playerDriven || this.driver !== null;
  }

  // --- Lifecycle (called by VehicleSystem) --------------------------------------

  attach(archetype: VehicleArchetype, position: V3, heading: number, color: number): void {
    this.entityId = allocEntityId();
    this.archetype = archetype;
    this.layout = carLayout(archetype.id);
    this.color = color;
    this.isAlive = true;
    this.playerDriven = false;
    this.driver = null;
    this.driverTypeId = null;
    this.lightsOverride = null;
    this.wheelSpin = 0;
    this.steerVisual = 0;
    this.visualPitch = 0;
    this.lastDamageByPlayer = false;
    this.fireTimer = 0;
    this.smokeFxTimer = 0;
    this.fireFxTimer = 0;
    this.wreckTimer = 0;
    this.hs.vf = this.hs.vl = this.hs.yawRate = 0;
    this.controls.throttle = this.controls.brake = this.controls.steer = 0;
    this.controls.handbrake = false;
    const s = this.state;
    s.speedMs = s.rpm01 = s.lateralSpeed = s.forwardSpeed = 0;
    s.health = archetype.health;
    s.onFire = s.exploded = s.wrecked = s.lightsOn = s.sirenOn = false;
    s.grounded = true;
    s.occupants.length = 0;
    this.seats.length = archetype.seats;
    this.seats.fill(null);
    this.cameraTarget.size = archetype.dims.length;
    this.velocity.set(0, 0, 0);
    this.prevLinvel.set(0, 0, 0);

    const physics = this.game.physics;
    const centerY = position.y + this.layout.colliderCenterY;
    this.body = physics.createDynamicBody(
      { x: position.x, y: centerY, z: position.z },
      { rotationY: heading, linearDamping: C.physics.linearDamping, angularDamping: C.physics.angularDamping, canSleep: true },
    );
    this.body.setEnabledRotations(false, true, false, true);
    const half = this.layout.colliderHalf;
    const desc = RAPIER.ColliderDesc.cuboid(half.x, half.y, half.z).setMass(archetype.mass).setRestitution(0.05);
    this.collider = physics.addCollider(desc, this.body, this, {
      groups: Groups.VEHICLE,
      filter: Groups.ALL,
      events: true,
      friction: C.physics.friction,
    });
    this.currPos.set(position.x, position.y, position.z);
    this.prevPos.copy(this.currPos);
    this.currHeading = this.prevHeading = wrapAngle(heading);
    this.object3d.visible = true;
    this.syncVisual(1);
  }

  detach(): void {
    this.driver?.dispose?.();
    this.driver = null;
    this.playerDriven = false;
    this.isAlive = false;
    this.object3d.visible = false;
    this.seats.fill(null);
    this.state.occupants.length = 0;
    this.game.physics.removeBody(this.body);
  }

  // --- Simulation ---------------------------------------------------------------

  fixedUpdate(dt: number): void {
    this.prevPos.copy(this.currPos);
    this.prevHeading = this.currHeading;
    this.readPose();
    const body = this.body;
    if (this.state.wrecked) {
      if (this.driver) this.setDriver(null);
      this.controls.throttle = this.controls.steer = 0;
      this.controls.brake = 1;
    }
    if (!this.hasController && body.isSleeping()) {
      this.velocity.set(0, 0, 0);
      this.prevLinvel.set(0, 0, 0);
      this.state.speedMs = this.state.forwardSpeed = this.state.lateralSpeed = 0;
      this.state.rpm01 = 0;
      return;
    }
    if (this.driver) this.driver.update(this, dt);
    else if (!this.playerDriven && !this.state.wrecked) {
      // Nobody at the wheel: coast with the parking brake creeping on so it does not roll forever
      // (a panicked driver leaves the brake fully applied).
      this.controls.throttle = this.controls.steer = 0;
      this.controls.brake = Math.max(0.25, this.controls.brake);
      this.controls.handbrake = false;
    }

    const lv = body.linvel();
    const h = this.currHeading;
    const fx = Math.sin(h);
    const fz = Math.cos(h);
    const rx = fz;
    const rz = -fx;
    this.hs.vf = lv.x * fx + lv.z * fz;
    this.hs.vl = lv.x * rx + lv.z * rz;
    const speed = Math.hypot(lv.x, lv.z);
    const c = this.controls;
    const active = this.hasController || speed > 0.05 || Math.abs(this.hs.yawRate) > 0.01;
    if (active) {
      this.state.grounded = this.probeGround();
      stepHandling(this.hs, this.archetype, c, dt, this.state.grounded);
      const wake = this.hasController;
      body.setLinvel({ x: fx * this.hs.vf + rx * this.hs.vl, y: lv.y, z: fz * this.hs.vf + rz * this.hs.vl }, wake);
      body.setAngvel({ x: 0, y: this.hs.yawRate, z: 0 }, wake);
    }
    const nv = body.linvel();
    this.velocity.set(nv.x, nv.y, nv.z);
    this.prevLinvel.copy(this.velocity);
    const s = this.state;
    s.forwardSpeed = this.hs.vf;
    s.lateralSpeed = this.hs.vl;
    s.speedMs = Math.hypot(nv.x, nv.z);
    s.rpm01 = this.hasController && !s.wrecked ? rpmFromSpeed(s.speedMs, this.archetype.topSpeed, c.throttle) : 0;
    this.wheelSpin += (this.hs.vf / this.archetype.dims.wheelRadius) * dt;
    this.steerVisual += (c.steer * C.handling.wheelSteerAngle - this.steerVisual) * Math.min(1, dt * 12);
  }

  private probeGround(): boolean {
    const t = this.body.translation();
    const len = this.layout.colliderHalf.y + C.physics.groundProbe;
    return this.game.physics.raycastInto(this.hit, t, DOWN, len, { groups: Groups.STATIC | Groups.VEHICLE, exclude: this }) !== null;
  }

  private readPose(): void {
    const t = this.body.translation();
    const r = this.body.rotation();
    this.currPos.set(t.x, t.y - this.layout.colliderCenterY, t.z);
    this.currHeading = wrapAngle(2 * Math.atan2(r.y, r.w));
  }

  // --- Visuals ------------------------------------------------------------------

  update(alpha: number): void {
    this.syncVisual(alpha);
  }

  private syncVisual(alpha: number): void {
    this.position.lerpVectors(this.prevPos, this.currPos, alpha);
    const h = lerpAngle(this.prevHeading, this.currHeading, alpha);
    headingQuaternion(h, this.quaternion);
    if (this.visualPitch !== 0) this.quaternion.multiply(tmpQuat.setFromAxisAngle(X_AXIS, this.visualPitch));
    this.object3d.position.copy(this.position);
    this.object3d.quaternion.copy(this.quaternion);
  }

  // --- Damage -------------------------------------------------------------------

  applyDamage(info: DamageInfo): void {
    if (!this.isAlive || this.state.wrecked) return;
    let amount = info.amount;
    if (info.type === 'bullet' && info.point) {
      // Rear third holds the gas tank.
      this.tmp.set(info.point.x - this.position.x, 0, info.point.z - this.position.z);
      const local = this.tmp.x * Math.sin(this.currHeading) + this.tmp.z * Math.cos(this.currHeading);
      if (local < -this.length / 6) amount *= C.damage.gasTankMultiplier;
    }
    const src = info.source;
    const playerEntity = this.game.systems.has('player') ? this.game.player.entity : null;
    if (src === 'player' || (src instanceof Vehicle && src.playerDriven) || (src !== undefined && src === playerEntity)) this.lastDamageByPlayer = true;
    else if (info.type !== 'fire') this.lastDamageByPlayer = false;
    this.state.health -= amount;
    this.game.events.emit('vehicle:damaged', { vehicle: this, amount, source: src ?? null });
    if (this.state.health <= 0 && !this.state.onFire && !this.state.exploded) this.ignite();
  }

  /** Health hit zero: burn for `fireSeconds`, then explode. */
  ignite(): void {
    this.state.onFire = true;
    this.fireTimer = C.damage.fireSeconds;
  }

  explode(): void {
    if (!this.isAlive || this.state.exploded) return;
    explodeVehicle(this.game, this);
  }

  // --- Enter / exit -------------------------------------------------------------

  canEnter(who: Entity): boolean {
    if (!this.isAlive || this.state.wrecked || this.state.onFire) return false;
    if (this.state.speedMs > C.interaction.maxEnterSpeed) return false;
    if (this.seats.includes(who)) return false;
    return this.seats.includes(null);
  }

  enter(who: Entity, seat = 0): number {
    if (!this.canEnter(who)) return -1;
    let idx = seat >= 0 && seat < this.seats.length && this.seats[seat] === null ? seat : this.seats.indexOf(null);
    if (idx < 0) return -1;
    this.seats[idx] = who;
    this.rebuildOccupants();
    if (idx === 0 && who.kind === 'player') {
      this.playerDriven = true;
      this.driverTypeId = null;
      this.body.wakeUp();
      this.body.enableCcd(true);
    }
    this.game.events.emit('vehicle:entered', { vehicle: this, who, seat: idx });
    return idx;
  }

  exit(who: Entity): void {
    const idx = this.seats.indexOf(who);
    if (idx < 0) return;
    this.seats[idx] = null;
    this.rebuildOccupants();
    if (who.kind === 'player') {
      this.playerDriven = false;
      this.body.enableCcd(false);
      this.controls.throttle = this.controls.steer = 0;
      this.controls.handbrake = false;
    }
    this.game.events.emit('vehicle:exited', { vehicle: this, who });
  }

  private rebuildOccupants(): void {
    const list = this.state.occupants;
    list.length = 0;
    for (const s of this.seats) if (s) list.push(s);
  }

  setDriver(driver: VehicleDriver | null): void {
    if (this.driver === driver) return;
    this.driver?.dispose?.();
    this.driver = driver;
    if (driver) this.body.wakeUp();
    else {
      this.controls.throttle = this.controls.steer = 0;
      this.controls.handbrake = false;
    }
  }

  seatDoorPosition(seat: number, out: Vector3): Vector3 {
    seatDoorLocal(this.archetype, seat, C.interaction.doorOutset, out);
    out.applyQuaternion(headingQuaternion(this.currHeading, tmpQuat)).add(this.currPos);
    out.y = this.currPos.y + 0.05;
    return out;
  }

  setSiren(on: boolean): void {
    if (!this.isPolice) return;
    this.state.sirenOn = on;
  }

  teleport(position: V3, heading: number): void {
    const centerY = position.y + this.layout.colliderCenterY;
    headingQuaternion(heading, tmpQuat);
    this.body.setTranslation({ x: position.x, y: centerY, z: position.z }, true);
    this.body.setRotation({ x: tmpQuat.x, y: tmpQuat.y, z: tmpQuat.z, w: tmpQuat.w }, true);
    this.body.setLinvel({ x: 0, y: 0, z: 0 }, true);
    this.body.setAngvel({ x: 0, y: 0, z: 0 }, true);
    this.hs.vf = this.hs.vl = this.hs.yawRate = 0;
    this.velocity.set(0, 0, 0);
    this.prevLinvel.set(0, 0, 0);
    this.currPos.set(position.x, position.y, position.z);
    this.prevPos.copy(this.currPos);
    this.currHeading = this.prevHeading = wrapAngle(heading);
    this.syncVisual(1);
  }

  /** World point above the hood where smoke and fire spawn. */
  hoodPosition(out: Vector3): Vector3 {
    return out.copy(this.layout.hoodPoint).applyQuaternion(this.quaternion).add(this.position);
  }
}

const X_AXIS = new Vector3(1, 0, 0);
const tmpQuat = new Quaternion();
