/**
 * Shared entity contracts. Every agent imports these from one place.
 *
 * Interfaces here are deliberately structural so that later agents' concrete
 * classes (Vehicle, Ped, Pickup...) can implement them without importing each
 * other. Extend open interfaces (PedContext, VehicleStateInfo, ...) from your
 * own module via declaration merging:
 *
 *   declare module '../../core/entities' {
 *     interface PedContext { peds: PedSystem }
 *   }
 */
import type { Object3D, Quaternion, Vector3 } from 'three';
import type { Game } from './Game';

/** Plain vector shape; THREE.Vector3 and Rapier vectors both satisfy it. */
export interface V3 {
  x: number;
  y: number;
  z: number;
}
export type Vector3Like = V3;

export type EntityKind =
  | 'player'
  | 'ped'
  | 'vehicle'
  | 'pickup'
  | 'prop'
  | 'building'
  | 'ground'
  | 'water'
  | 'trigger'
  | 'interior'
  | 'other';

export interface Entity {
  readonly id: number;
  readonly kind: EntityKind;
  /** Root object in the scene (null for invisible entities such as trigger volumes). */
  object3d: Object3D | null;
  /** World position; read-only for outsiders. Systems update it every frame. */
  readonly position: Vector3;
  isAlive: boolean;
}

export type DamageType = 'bullet' | 'melee' | 'explosion' | 'vehicle' | 'fall' | 'fire';
export type BodyPart = 'head' | 'body' | 'limb';

export interface DamageInfo {
  amount: number;
  type: DamageType;
  point?: V3;
  direction?: V3;
  part?: BodyPart;
  source?: Entity | 'player';
}

export interface IDamageable {
  applyDamage(info: DamageInfo): void;
}

export function isDamageable(e: unknown): e is IDamageable {
  return !!e && typeof (e as IDamageable).applyDamage === 'function';
}

export interface IEnterable {
  canEnter(who: Entity): boolean;
  /** Returns the seat index the occupant took, or -1 if refused. */
  enter(who: Entity, seat?: number): number;
  exit(who: Entity): void;
}

export function isEnterable(e: unknown): e is IEnterable {
  const c = e as IEnterable;
  return !!c && typeof c.canEnter === 'function' && typeof c.enter === 'function';
}

/** What the camera follows. Player and vehicles expose one of these. */
export interface CameraTarget {
  position: V3;
  quaternion: Quaternion;
  velocity: V3;
  kind: 'onFoot' | 'vehicle';
  /** Approximate longest dimension in meters; vehicle camera distance scales with it. */
  size?: number;
}

// ---------------------------------------------------------------------------
// Vehicle plugin point (Agent 2 implements Vehicle; drivers plug into it)
// ---------------------------------------------------------------------------

export interface VehicleControls {
  throttle: number; // 0..1
  brake: number; // 0..1
  steer: number; // -1..1 (left negative)
  handbrake: boolean;
}

/** Open interface: Agent 2 may merge extra fields. */
export interface VehicleStateInfo {
  speedMs: number;
  rpm01: number;
  health: number;
  onFire: boolean;
  exploded: boolean;
  occupants: Entity[];
}

export interface IVehicle extends Entity, IEnterable, IDamageable {
  readonly archetypeId: string;
  controls: VehicleControls;
  readonly state: VehicleStateInfo;
  driver: VehicleDriver | null;
  cameraTarget: CameraTarget;
  /** World linear velocity (m/s). */
  readonly velocity: Vector3;
  /** Yaw in radians; 0 faces +Z (same convention as peds). */
  heading: number;
  /** Body color (sRGB hex). */
  color: number;
  readonly isPolice: boolean;
  /** Approximate length in meters (for camera distance and parking). */
  readonly length: number;
  /** Occupant in seat 0, if any. */
  readonly driverEntity: Entity | null;
  setDriver(driver: VehicleDriver | null): void;
  /** World position of a seat's door (for exit / enter animations). */
  seatDoorPosition(seat: number, out: Vector3): Vector3;
  /** Immediately destroys the vehicle (fireball, radius damage). */
  explode(): void;
  /** Police light bar / siren on-off; no-op on civilian archetypes. */
  setSiren(on: boolean): void;
  /** Teleports the body and zeroes velocity. */
  teleport(position: V3, heading: number): void;
}

export interface VehicleDriver {
  update(vehicle: IVehicle, dt: number): void;
  dispose?(): void;
}

// ---------------------------------------------------------------------------
// Ped plugin point (Agent 3 implements Ped; brains plug into it)
// ---------------------------------------------------------------------------

export type PedState =
  | 'wander'
  | 'idleGroup'
  | 'cower'
  | 'flee'
  | 'callPolice'
  | 'fight'
  | 'knockedDown'
  | 'dead'
  | 'inVehicle'
  | 'scripted';

/** Open interface: extended by agents via declaration merging. */
export interface PedContext {
  game: Game;
  playerPosition: V3;
  /** Horizontal distance from this ped to the player, refreshed before each brain update. */
  playerDistance: number;
}

export interface IPed extends Entity, IDamageable {
  readonly typeId: string;
  state: PedState;
  brain: PedBrain;
  heading: number;
  health: number;
  /** Horizontal velocity (m/s). */
  readonly velocity: Vector3;
  /** Weapon this ped fires through `weapons.npcFire`, if any. */
  weaponId: string | null;
  /** Cower pose flag; brains set it, the renderer reads it. */
  handsUp: boolean;
  /** True for cop and SWAT ped types (wanted system witness logic). */
  readonly isCop: boolean;
  /** Vehicle this ped is sitting in, if any. */
  vehicle: IVehicle | null;
  setBrain(brain: PedBrain): void;
  // --- Motion primitives for brains (applied in the ped's fixedUpdate) ---
  /** Walk/run toward a world point at `speedMs`; stops within 0.3 m. */
  moveToward(target: V3, speedMs: number): void;
  stop(): void;
  /** Turn to face a world point (does not move). */
  face(target: V3): void;
  /** Swap to a dynamic body and apply an impulse (world units, kg·m/s). */
  ragdoll(impulse: V3): void;
  enterVehicle(vehicle: IVehicle, seat?: number): boolean;
  exitVehicle(): void;
  /** Kills the ped outright with attribution. */
  kill(source?: Entity | 'player'): void;
}

export interface PedBrain {
  update(ped: IPed, ctx: PedContext, dt: number): void;
  onEvent?(ped: IPed, event: PedEvent): void;
}

/** Events a ped brain can react to. Agents may merge more variants via `PedEventMap`. */
export interface PedEventMap {
  damaged: DamageInfo;
  aimedAt: { seconds: number };
  heardGunshot: { position: V3; distance: number };
  hitByVehicle: { impulse: number };
}
export type PedEvent = { [K in keyof PedEventMap]: { type: K } & PedEventMap[K] }[keyof PedEventMap];

// ---------------------------------------------------------------------------
// Id allocation
// ---------------------------------------------------------------------------

let nextEntityId = 1;
/** Monotonic ids for all entities across systems. */
export function allocEntityId(): number {
  return nextEntityId++;
}
