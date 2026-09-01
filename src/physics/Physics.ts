import RAPIER, { type Collider, type RigidBody, type World } from '@dimforge/rapier3d-compat';
import { Quaternion, Vector3 } from 'three';
import { Config } from '../core/Config';
import type { Entity, V3 } from '../core/entities';
import type { Events } from '../core/Events';

/**
 * Collision group bits. A collider's interaction groups are
 * `interactionGroups(membership, filter)`; two colliders interact when each one's
 * membership intersects the other's filter.
 */
export const Groups = {
  STATIC: 1 << 0,
  PLAYER: 1 << 1,
  VEHICLE: 1 << 2,
  PED: 1 << 3,
  PICKUP: 1 << 4,
  TRIGGER: 1 << 5,
  BULLET_BLOCKER: 1 << 6,
  ALL: 0xffff,
} as const;

export function interactionGroups(membership: number, filter: number = Groups.ALL): number {
  return ((membership & 0xffff) << 16) | (filter & 0xffff);
}

export interface RaycastHit {
  entity: Entity | null;
  point: Vector3;
  normal: Vector3;
  toi: number;
  collider: Collider;
}

export interface RaycastOptions {
  /** Only hit colliders whose membership intersects these bits (default: everything). */
  groups?: number;
  /** Collider, body or entity to ignore (typically the shooter). */
  exclude?: Collider | RigidBody | Entity;
  /** Treat rays starting inside a shape as hitting at toi 0 (default true). */
  solid?: boolean;
  /** Skip sensor colliders (default true). */
  excludeSensors?: boolean;
}

export interface StaticColliderOptions {
  entity?: Entity;
  rotationY?: number;
  rotation?: { x: number; y: number; z: number; w: number };
  groups?: number;
  filter?: number;
  sensor?: boolean;
  events?: boolean;
  friction?: number;
}

export interface BodyOptions {
  rotationY?: number;
  linearDamping?: number;
  angularDamping?: number;
  ccd?: boolean;
  canSleep?: boolean;
}

/**
 * Rapier wrapper: entity registry per collider, raycasts that resolve to entities,
 * contact/intersection events fanned out through `game.events`.
 */
export class Physics {
  readonly world: World;
  private readonly eventQueue: RAPIER.EventQueue;
  private readonly entityByCollider = new Map<number, Entity>();
  private readonly collidersByEntity = new Map<Entity, Collider[]>();
  private readonly ray = new RAPIER.Ray({ x: 0, y: 0, z: 0 }, { x: 0, y: -1, z: 0 });
  private readonly tmpQuat = new Quaternion();

  constructor(private readonly events: Events) {
    this.world = new RAPIER.World({ x: 0, y: Config.physics.gravity, z: 0 });
    this.world.timestep = Config.loop.fixedDt;
    this.eventQueue = new RAPIER.EventQueue(true);
  }

  // --- Registry --------------------------------------------------------------

  register(collider: Collider, entity: Entity): void {
    this.entityByCollider.set(collider.handle, entity);
    let list = this.collidersByEntity.get(entity);
    if (!list) {
      list = [];
      this.collidersByEntity.set(entity, list);
    }
    if (!list.includes(collider)) list.push(collider);
  }

  unregister(collider: Collider): void {
    const entity = this.entityByCollider.get(collider.handle);
    this.entityByCollider.delete(collider.handle);
    if (!entity) return;
    const list = this.collidersByEntity.get(entity);
    if (!list) return;
    const idx = list.indexOf(collider);
    if (idx >= 0) list.splice(idx, 1);
    if (list.length === 0) this.collidersByEntity.delete(entity);
  }

  entityOf(colliderHandle: number): Entity | null {
    return this.entityByCollider.get(colliderHandle) ?? null;
  }

  entityOfCollider(collider: Collider): Entity | null {
    return this.entityByCollider.get(collider.handle) ?? null;
  }

  entityOfBody(body: RigidBody): Entity | null {
    for (let i = 0; i < body.numColliders(); i++) {
      const e = this.entityByCollider.get(body.collider(i).handle);
      if (e) return e;
    }
    return null;
  }

  collidersOf(entity: Entity): readonly Collider[] {
    return this.collidersByEntity.get(entity) ?? [];
  }

  // --- Queries ---------------------------------------------------------------

  /** Allocating raycast; use `raycastInto` in per-frame paths. */
  raycast(origin: V3, dir: V3, maxToi: number, opts?: RaycastOptions): RaycastHit | null {
    const out: RaycastHit = {
      entity: null,
      point: new Vector3(),
      normal: new Vector3(),
      toi: 0,
      collider: null as unknown as Collider,
    };
    return this.raycastInto(out, origin, dir, maxToi, opts);
  }

  /** Non-allocating raycast: fills `out` and returns it, or null on miss. `dir` must be normalised. */
  raycastInto(out: RaycastHit, origin: V3, dir: V3, maxToi: number, opts?: RaycastOptions): RaycastHit | null {
    this.ray.origin.x = origin.x;
    this.ray.origin.y = origin.y;
    this.ray.origin.z = origin.z;
    this.ray.dir.x = dir.x;
    this.ray.dir.y = dir.y;
    this.ray.dir.z = dir.z;
    const solid = opts?.solid ?? true;
    const excludeSensors = opts?.excludeSensors ?? true;
    const groups = interactionGroups(Groups.ALL, opts?.groups ?? Groups.ALL);
    let excludeCollider: Collider | undefined;
    let excludeBody: RigidBody | undefined;
    let predicate: ((c: Collider) => boolean) | undefined;
    const ex = opts?.exclude;
    if (ex) {
      if (ex instanceof RAPIER.Collider) excludeCollider = ex;
      else if (ex instanceof RAPIER.RigidBody) excludeBody = ex;
      else {
        const list = this.collidersByEntity.get(ex);
        if (list && list.length === 1) excludeCollider = list[0];
        else if (list) predicate = (c) => !list.includes(c) && (!excludeSensors || !c.isSensor());
      }
    }
    if (!predicate && excludeSensors) predicate = (c) => !c.isSensor();
    const flags = excludeSensors ? RAPIER.QueryFilterFlags.EXCLUDE_SENSORS : undefined;
    const hit = this.world.castRayAndGetNormal(this.ray, maxToi, solid, flags, groups, excludeCollider, excludeBody, predicate);
    if (!hit) return null;
    out.toi = hit.timeOfImpact;
    out.collider = hit.collider;
    out.entity = this.entityByCollider.get(hit.collider.handle) ?? null;
    out.point.set(origin.x + dir.x * hit.timeOfImpact, origin.y + dir.y * hit.timeOfImpact, origin.z + dir.z * hit.timeOfImpact);
    out.normal.set(hit.normal.x, hit.normal.y, hit.normal.z);
    return out;
  }

  /** Ground height below a point (static geometry only), or null if nothing within `maxDrop`. */
  groundHeightAt(x: number, z: number, fromY = 200, maxDrop = 400): number | null {
    this.ray.origin.x = x;
    this.ray.origin.y = fromY;
    this.ray.origin.z = z;
    this.ray.dir.x = 0;
    this.ray.dir.y = -1;
    this.ray.dir.z = 0;
    const hit = this.world.castRay(
      this.ray,
      maxDrop,
      true,
      RAPIER.QueryFilterFlags.EXCLUDE_SENSORS,
      interactionGroups(Groups.ALL, Groups.STATIC),
    );
    return hit ? fromY - hit.timeOfImpact : null;
  }

  // --- Creation helpers -----------------------------------------------------

  private applyColliderOptions(desc: RAPIER.ColliderDesc, opts?: StaticColliderOptions): RAPIER.ColliderDesc {
    const membership = opts?.groups ?? Groups.STATIC;
    desc.setCollisionGroups(interactionGroups(membership, opts?.filter ?? Groups.ALL));
    if (opts?.sensor) desc.setSensor(true);
    if (opts?.events || opts?.sensor) desc.setActiveEvents(RAPIER.ActiveEvents.COLLISION_EVENTS);
    if (opts?.friction !== undefined) desc.setFriction(opts.friction);
    if (opts?.rotation) desc.setRotation(opts.rotation);
    else if (opts?.rotationY) {
      this.tmpQuat.setFromAxisAngle(UP, opts.rotationY);
      desc.setRotation({ x: this.tmpQuat.x, y: this.tmpQuat.y, z: this.tmpQuat.z, w: this.tmpQuat.w });
    }
    return desc;
  }

  /** Static box (no rigid body). `halfExtents` are half sizes. */
  addStaticCuboid(center: V3, halfExtents: V3, opts?: StaticColliderOptions): Collider {
    const desc = RAPIER.ColliderDesc.cuboid(halfExtents.x, halfExtents.y, halfExtents.z).setTranslation(center.x, center.y, center.z);
    this.applyColliderOptions(desc, opts);
    const collider = this.world.createCollider(desc);
    if (opts?.entity) this.register(collider, opts.entity);
    return collider;
  }

  addStaticTrimesh(vertices: Float32Array, indices: Uint32Array, opts?: StaticColliderOptions & { position?: V3 }): Collider {
    const desc = RAPIER.ColliderDesc.trimesh(vertices, indices);
    if (opts?.position) desc.setTranslation(opts.position.x, opts.position.y, opts.position.z);
    this.applyColliderOptions(desc, opts);
    const collider = this.world.createCollider(desc);
    if (opts?.entity) this.register(collider, opts.entity);
    return collider;
  }

  createKinematicBody(position: V3, opts?: BodyOptions): RigidBody {
    const desc = RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(position.x, position.y, position.z);
    if (opts?.rotationY) {
      this.tmpQuat.setFromAxisAngle(UP, opts.rotationY);
      desc.setRotation({ x: this.tmpQuat.x, y: this.tmpQuat.y, z: this.tmpQuat.z, w: this.tmpQuat.w });
    }
    return this.world.createRigidBody(desc);
  }

  createDynamicBody(position: V3, opts?: BodyOptions): RigidBody {
    const desc = RAPIER.RigidBodyDesc.dynamic().setTranslation(position.x, position.y, position.z);
    if (opts?.rotationY) {
      this.tmpQuat.setFromAxisAngle(UP, opts.rotationY);
      desc.setRotation({ x: this.tmpQuat.x, y: this.tmpQuat.y, z: this.tmpQuat.z, w: this.tmpQuat.w });
    }
    if (opts?.linearDamping !== undefined) desc.setLinearDamping(opts.linearDamping);
    if (opts?.angularDamping !== undefined) desc.setAngularDamping(opts.angularDamping);
    if (opts?.ccd) desc.setCcdEnabled(true);
    if (opts?.canSleep !== undefined) desc.setCanSleep(opts.canSleep);
    return this.world.createRigidBody(desc);
  }

  /** Attach a collider to a body and register its entity. */
  addCollider(desc: RAPIER.ColliderDesc, body: RigidBody, entity?: Entity, opts?: StaticColliderOptions): Collider {
    this.applyColliderOptions(desc, { groups: Groups.STATIC, ...opts });
    const collider = this.world.createCollider(desc, body);
    if (entity) this.register(collider, entity);
    return collider;
  }

  removeCollider(collider: Collider): void {
    this.unregister(collider);
    this.world.removeCollider(collider, true);
  }

  removeBody(body: RigidBody): void {
    for (let i = 0; i < body.numColliders(); i++) this.unregister(body.collider(i));
    this.world.removeRigidBody(body);
  }

  // --- Stepping --------------------------------------------------------------

  step(dt: number): void {
    this.world.timestep = dt;
    this.world.step(this.eventQueue);
    this.eventQueue.drainCollisionEvents((h1, h2, started) => {
      const payload = {
        a: this.entityByCollider.get(h1) ?? null,
        b: this.entityByCollider.get(h2) ?? null,
        colliderA: h1,
        colliderB: h2,
      };
      this.events.emit(started ? 'physics:contactStarted' : 'physics:contactEnded', payload);
    });
  }

  dispose(): void {
    this.entityByCollider.clear();
    this.collidersByEntity.clear();
    this.eventQueue.free();
    this.world.free();
  }
}

const UP = new Vector3(0, 1, 0);
export { RAPIER };
