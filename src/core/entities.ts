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
}

export interface IPed extends Entity, IDamageable {
  readonly typeId: string;
  state: PedState;
  brain: PedBrain;
  heading: number;
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
