import { Vector3 } from 'three';
import type { IVehicle, V3, VehicleDriver } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { IVehiclesPeer, VehicleArchetypeId } from '../../core/peers';
import { Pool } from '../../core/Pool';
import { Random } from '../../core/Random';
import { SpatialHash } from '../../core/SpatialHash';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { isVehicleArchetypeId, VEHICLES } from '../../data/vehicles';
import { Vehicle } from '../../entities/Vehicle';
import { Groups, type RaycastHit } from '../../physics/Physics';
import { VehicleConfig } from './config';
import './events';
import { pickColor, spawnParkedCars } from './ParkedSpawner';
import { PlayerControl } from './PlayerControl';
import { VehicleAudio } from './VehicleAudio';
import { handleVehicleContact, tickVehicleFx } from './VehicleFx';
import { VehicleRenderer } from './VehicleRenderer';

const C = VehicleConfig;
const DOWN = { x: 0, y: -1, z: 0 };

interface SavedVehicle {
  id: VehicleArchetypeId;
  x: number;
  y: number;
  z: number;
  heading: number;
  color: number;
  health: number;
}

/**
 * Owns every Vehicle (parked, traffic, player), the shared instanced renderer, the
 * vehicle spatial hash, collision damage, fire/explosions and the player's enter/exit.
 * Traffic drivers and spawning live in `systems/traffic`.
 */
export class VehicleSystem implements GameSystem, IVehiclesPeer {
  private readonly live: Vehicle[] = [];
  private readonly pool: Pool<Vehicle>;
  readonly hash = new SpatialHash<Vehicle>(20);
  private renderer!: VehicleRenderer;
  private readonly audio: VehicleAudio;
  readonly control: PlayerControl;
  private readonly rng = Random.unseeded();
  private readonly scratch: Vehicle[] = [];
  private readonly tmp = new Vector3();
  private readonly tmp2 = new Vector3();
  private readonly hit: RaycastHit = { entity: null, point: new Vector3(), normal: new Vector3(), toi: 0, collider: null as unknown as RaycastHit['collider'] };
  private wreckSweep = 0;
  private offContact: (() => void) | null = null;

  constructor(private readonly game: Game) {
    this.pool = new Pool<Vehicle>(() => new Vehicle(game));
    this.audio = new VehicleAudio(game);
    this.control = new PlayerControl(game, (pos, radius) => this.nearestEnterableVehicle(pos, radius));
  }

  init(): void {
    this.renderer = new VehicleRenderer(this.game.worldRoot, C.maxVehicles);
    this.offContact = this.game.events.on('physics:contactStarted', (e) => {
      if (e.a instanceof Vehicle) handleVehicleContact(this.game, e.a, e.b);
      if (e.b instanceof Vehicle) handleVehicleContact(this.game, e.b, e.a);
    });
    const parked = spawnParkedCars(this.game, this, Random.fromString('leonida-parked'));
    this.game.perf.set('vehicles', this.live.length);
    console.info(`[vehicles] ${parked} parked cars spawned`);
  }

  // --- IVehiclesPeer -------------------------------------------------------------

  spawn(
    archetypeId: VehicleArchetypeId,
    position: V3,
    heading: number,
    opts?: { color?: number; driver?: VehicleDriver | null; parked?: boolean },
  ): Vehicle {
    if (this.renderer.freeCount === 0 && !this.evictOne()) throw new Error('[vehicles] no free vehicle slots');
    const archetype = VEHICLES[archetypeId];
    const v = this.pool.acquire();
    v.attach(archetype, position, heading, opts?.color ?? pickColor(archetypeId, this.rng));
    v.slot = this.renderer.alloc();
    if (opts?.driver) v.setDriver(opts.driver);
    if (opts?.parked) v.body.sleep();
    this.live.push(v);
    this.hash.insert(v, position.x, position.z);
    this.game.events.emit('vehicle:spawned', { vehicle: v });
    return v;
  }

  despawn(vehicle: IVehicle): void {
    const v = vehicle as Vehicle;
    if (!(v instanceof Vehicle) || !v.isAlive) return;
    this.control.forceExit(v);
    this.audio.remove(v);
    this.hash.remove(v);
    this.renderer.free(v.slot);
    v.slot = -1;
    const idx = this.live.indexOf(v);
    if (idx >= 0) {
      this.live[idx] = this.live[this.live.length - 1];
      this.live.pop();
    }
    this.game.events.emit('vehicle:despawned', { vehicle: v });
    v.detach();
    this.pool.release(v);
  }

  /** Frees a slot by recycling the farthest idle car (never the player's or an occupied one). */
  private evictOne(): boolean {
    const ref = this.game.systems.has('player') ? this.game.player.position : this.tmp.set(0, 0, 0);
    let best: Vehicle | null = null;
    let bestD = -1;
    for (const v of this.live) {
      if (v.hasController || v.state.occupants.length > 0) continue;
      const d = v.position.distanceToSquared(ref);
      if (d > bestD) {
        bestD = d;
        best = v;
      }
    }
    if (!best) return false;
    this.despawn(best);
    return true;
  }

  nearestEnterable(pos: V3, radius: number): IVehicle | null {
    return this.nearestEnterableVehicle(pos, radius);
  }

  private nearestEnterableVehicle(pos: V3, radius: number): Vehicle | null {
    if (!this.game.systems.has('player')) return null;
    const who = this.game.player.entity;
    this.hash.query(pos.x, pos.z, radius + 3, this.scratch);
    let best: Vehicle | null = null;
    let bestD = Infinity;
    for (let i = 0; i < this.scratch.length; i++) {
      const v = this.scratch[i];
      if (!v.canEnter(who)) continue;
      const d = Math.hypot(v.position.x - pos.x, v.position.z - pos.z) - v.length / 2;
      if (d <= radius && d < bestD) {
        bestD = d;
        best = v;
      }
    }
    return best;
  }

  query(pos: V3, radius: number, out: IVehicle[]): IVehicle[] {
    // The hash only ever holds Vehicles, so widening the output array is safe.
    return this.hash.query(pos.x, pos.z, radius, out as Vehicle[]);
  }

  get all(): readonly IVehicle[] {
    return this.live;
  }

  get playerVehicle(): Vehicle | null {
    return this.control.vehicle;
  }

  enterNearest(): IVehicle | null {
    return this.control.enterNearest();
  }

  exitCurrent(): void {
    this.control.exitCurrent();
  }

  // --- Loop ------------------------------------------------------------------------

  fixedUpdate(dt: number): void {
    this.control.fixedUpdate(dt);
    const night = this.game.systems.has('world') ? this.game.world.night01 : 0;
    const autoLights = night > C.lights.autoOnNight;
    const live = this.live;
    for (let i = 0; i < live.length; i++) {
      const v = live[i];
      v.state.lightsOn = v.hasController && !v.state.wrecked && (v.lightsOverride ?? autoLights);
      v.fixedUpdate(dt);
      if (!v.body.isSleeping()) this.hash.update(v, v.position.x, v.position.z);
      tickVehicleFx(this.game, v, dt);
    }
    this.game.perf.set('vehicles', live.length);
  }

  update(dt: number, alpha: number): void {
    const cam = this.game.viewCamera.position;
    const night = this.game.systems.has('world') ? this.game.world.night01 : 0;
    const t = this.game.time.realElapsed;
    const r2 = C.renderDistance * C.renderDistance;
    const player = this.control.vehicle;
    if (player) this.updateVisualPitch(player, dt);
    const live = this.live;
    for (let i = 0; i < live.length; i++) {
      const v = live[i];
      v.update(alpha);
      if (v.position.distanceToSquared(cam) < r2) this.renderer.write(v, night, t);
      else this.renderer.hide(v.slot);
    }
    this.renderer.commit();
    const listener = this.game.systems.has('player') ? this.game.player.position : cam;
    this.audio.update(live, listener);
    this.control.update();

    this.wreckSweep += dt;
    if (this.wreckSweep > 1) {
      this.wreckSweep = 0;
      this.sweepWrecks(listener);
    }
  }

  /** Tilt the player's car to the road under its axles (bridge ramps) without touching physics. */
  private updateVisualPitch(v: Vehicle, dt: number): void {
    const half = v.archetype.dims.wheelbase / 2;
    const fx = Math.sin(v.heading) * half;
    const fz = Math.cos(v.heading) * half;
    const yFront = this.probeY(v.position.x + fx, v.position.y, v.position.z + fz, v);
    const yRear = this.probeY(v.position.x - fx, v.position.y, v.position.z - fz, v);
    const target = yFront !== null && yRear !== null ? -Math.atan2(yFront - yRear, half * 2) : 0;
    v.visualPitch += (target - v.visualPitch) * Math.min(1, dt * 8);
  }

  private probeY(x: number, y: number, z: number, exclude: Vehicle): number | null {
    this.tmp2.set(x, y + 1.5, z);
    const hit = this.game.physics.raycastInto(this.hit, this.tmp2, DOWN, 4, { groups: Groups.STATIC, exclude });
    return hit ? hit.point.y : null;
  }

  private sweepWrecks(ref: V3): void {
    const d2 = C.damage.wreckDespawnDistance ** 2;
    for (let i = this.live.length - 1; i >= 0; i--) {
      const v = this.live[i];
      if (!v.state.exploded || v.wreckTimer < C.damage.wreckDespawnSeconds) continue;
      if (v.position.distanceToSquared(ref) > d2) this.despawn(v);
    }
  }

  // --- Save ----------------------------------------------------------------------

  serialize(): { vehicle: SavedVehicle | null } {
    const v = this.control.vehicle;
    if (!v) return { vehicle: null };
    return {
      vehicle: { id: v.archetypeId, x: v.position.x, y: v.position.y, z: v.position.z, heading: v.heading, color: v.color, health: v.state.health },
    };
  }

  deserialize(data: unknown): void {
    const d = data as { vehicle?: SavedVehicle | null } | null;
    const s = d?.vehicle;
    if (!s || !isVehicleArchetypeId(s.id) || !this.game.systems.has('player')) return;
    this.control.exitCurrent();
    // Restore the car beside the player (who was inside it when saved) rather than on top of them.
    const p = this.game.player.position;
    const right = this.tmp.set(Math.cos(s.heading), 0, -Math.sin(s.heading)).multiplyScalar(3);
    const ground = this.game.physics.groundHeightAt(p.x + right.x, p.z + right.z) ?? p.y;
    const v = this.spawn(s.id, { x: p.x + right.x, y: ground + 0.05, z: p.z + right.z }, s.heading, { color: s.color });
    v.state.health = Math.max(1, s.health);
  }

  dispose(): void {
    this.offContact?.();
    for (let i = this.live.length - 1; i >= 0; i--) this.despawn(this.live[i]);
    this.audio.dispose();
    this.control.dispose();
    this.renderer.dispose(this.game.worldRoot);
  }

  // --- Debug -----------------------------------------------------------------------

  private groundY(x: number, z: number, fallback: number): number {
    return (this.game.physics.groundHeightAt(x, z) ?? fallback) + 0.05;
  }

  debugActions = {
    spawn: (archetype: unknown, x: unknown, z: unknown, heading?: unknown): number => {
      const id = String(archetype);
      if (!isVehicleArchetypeId(id)) throw new Error(`unknown archetype '${id}'`);
      const px = Number(x);
      const pz = Number(z);
      return this.spawn(id, { x: px, y: this.groundY(px, pz, 0), z: pz }, Number(heading ?? 0)).id;
    },
    spawnHere: (archetype: unknown): number => {
      const id = String(archetype);
      if (!isVehicleArchetypeId(id)) throw new Error(`unknown archetype '${id}'`);
      const p = this.game.player.position;
      const h = this.game.player.heading;
      const x = p.x + Math.sin(h) * 5;
      const z = p.z + Math.cos(h) * 5;
      return this.spawn(id, { x, y: this.groundY(x, z, p.y), z }, h).id;
    },
    enterNearest: (): number | null => this.control.enterNearest()?.id ?? null,
    exit: (): void => this.control.exitCurrent(),
    explodeCurrent: (): boolean => {
      const v = this.control.vehicle;
      if (!v) return false;
      v.lastDamageByPlayer = true;
      v.explode();
      return true;
    },
    count: (): number => this.live.length,
    state: (): Record<string, unknown> | null => {
      const v = this.control.vehicle;
      if (!v) return null;
      return {
        id: v.id,
        archetype: v.archetypeId,
        speedMs: +v.state.speedMs.toFixed(2),
        forward: +v.state.forwardSpeed.toFixed(2),
        lateral: +v.state.lateralSpeed.toFixed(2),
        yawRate: +v.hs.yawRate.toFixed(3),
        health: Math.round(v.state.health),
        onFire: v.state.onFire,
        exploded: v.state.exploded,
        grounded: v.state.grounded,
        position: { x: +v.position.x.toFixed(2), y: +v.position.y.toFixed(2), z: +v.position.z.toFixed(2) },
        heading: +v.heading.toFixed(3),
      };
    },
    despawnAll: (): number => {
      const n = this.live.length;
      for (let i = this.live.length - 1; i >= 0; i--) if (!this.live[i].playerDriven) this.despawn(this.live[i]);
      return n - this.live.length;
    },
  };
}

const module: SystemModule = {
  id: 'vehicles',
  priority: PRIORITY.vehicles,
  create: (game) => new VehicleSystem(game),
};

export default module;
