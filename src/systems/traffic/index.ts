import { Frustum, Matrix4, Vector3 } from 'three';
import type { IVehicle, V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { ITrafficPeer, IVehiclesPeer } from '../../core/peers';
import { Random } from '../../core/Random';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { TRAFFIC_DRIVER_TYPES } from '../../data/vehicles';
import { Vehicle } from '../../entities/Vehicle';
import type { RoadGraph } from '../../world/RoadGraph';
import { pickArchetypeFor, pickColor } from '../vehicles/ParkedSpawner';
import { TrafficConfig } from './config';
import { TrafficDriver, type TrafficContext } from './TrafficDriver';
import { TrafficLights } from './TrafficLights';

const T = TrafficConfig;

/**
 * Keeps ~`density` driven cars around the player: spawns on lane points outside the
 * camera frustum, recycles far/stuck cars, runs the traffic light clock, and makes
 * drivers bail out (panic) near gunfire and explosions.
 */
export class TrafficSystem implements GameSystem, ITrafficPeer, TrafficContext {
  /** Cars with a TrafficDriver at the wheel. */
  private readonly cars: Vehicle[] = [];
  /** Cars whose driver bailed or was carjacked; still ours to recycle when far away. */
  private readonly abandoned: Vehicle[] = [];
  private target: number = T.density;
  lights!: TrafficLights;
  graph!: RoadGraph;
  readonly rng = Random.unseeded();
  private vehicles: IVehiclesPeer | null = null;
  private readonly frustum = new Frustum();
  private readonly projView = new Matrix4();
  private readonly scratch: IVehicle[] = [];
  private readonly door = new Vector3();
  private spawnTimer = 0;
  private readonly offs: (() => void)[] = [];

  constructor(private readonly game: Game) {}

  init(): void {
    const g = this.game;
    if (!g.systems.has('world') || !g.vehicles) {
      console.warn('[traffic] needs the world and vehicles systems; traffic disabled');
      return;
    }
    this.vehicles = g.vehicles;
    this.graph = g.world.roadGraph;
    this.lights = new TrafficLights(this.graph, g.world.city.props);
    this.offs.push(
      g.events.on('crime:committed', (e) => {
        if (e.loud) this.panicNear(e.position, T.panic.crimeRadius);
      }),
      g.events.on('vehicle:exploded', (e) => this.panicNear(e.position, T.panic.explosionRadius)),
      g.events.on('vehicle:despawned', (e) => {
        remove(this.cars, e.vehicle);
        remove(this.abandoned, e.vehicle);
      }),
      g.events.on('vehicle:carjacked', (e) => this.release(e.vehicle)),
      g.events.on('vehicle:entered', (e) => {
        if (e.who.kind === 'player') this.release(e.vehicle);
      }),
    );
  }

  // --- ITrafficPeer ----------------------------------------------------------------

  setDensity(count: number): void {
    this.target = Math.max(0, Math.floor(count));
  }

  get density(): number {
    return this.target;
  }

  get count(): number {
    return this.cars.length;
  }

  isGreen(nodeId: number, edgeId: number): boolean {
    return this.lights ? this.lights.isGreen(nodeId, edgeId) : true;
  }

  panicNear(pos: V3, radius: number): void {
    for (let i = this.cars.length - 1; i >= 0; i--) {
      const car = this.cars[i];
      const dx = car.position.x - pos.x;
      const dz = car.position.z - pos.z;
      if (dx * dx + dz * dz <= radius * radius) this.bail(car);
    }
  }

  // --- TrafficContext --------------------------------------------------------------

  queryVehicles(pos: V3, radius: number): readonly IVehicle[] {
    return this.vehicles ? this.vehicles.query(pos, radius, this.scratch) : this.scratch;
  }

  playerOnFoot(): V3 | null {
    if (!this.game.systems.has('player')) return null;
    const p = this.game.player;
    return p.isMounted || !p.isAlive ? null : p.position;
  }

  honk(vehicle: IVehicle): void {
    const v = vehicle as Vehicle;
    this.game.audio.play(v.archetype.hornSound, { position: v.position });
    this.game.events.emit('vehicle:horn', { vehicle });
  }

  // --- Loop --------------------------------------------------------------------------

  fixedUpdate(dt: number): void {
    if (!this.vehicles) return;
    this.lights.update(dt);
    const ref = this.game.systems.has('player') ? this.game.player.position : this.game.viewCamera.position;
    const despawn2 = T.despawnDistance * T.despawnDistance;

    for (let i = this.cars.length - 1; i >= 0; i--) {
      const car = this.cars[i];
      if (!car.isAlive) {
        this.cars.splice(i, 1);
        continue;
      }
      if (car.state.onFire || car.state.exploded) {
        this.bail(car);
        continue;
      }
      if (!(car.driver instanceof TrafficDriver)) {
        // Someone else took the wheel (carjack, police, scripts): no longer ours to drive.
        this.cars.splice(i, 1);
        if (car.driver === null) this.abandoned.push(car);
        continue;
      }
      const d2 = car.position.distanceToSquared(ref);
      if (d2 > despawn2 || (car.driver.wantsDespawn && !this.frustum.containsPoint(car.position))) {
        this.vehicles.despawn(car);
      }
    }
    for (let i = this.abandoned.length - 1; i >= 0; i--) {
      const car = this.abandoned[i];
      if (!car.isAlive) {
        this.abandoned.splice(i, 1);
        continue;
      }
      if (car.playerDriven || car.state.occupants.length > 0) continue;
      if (car.position.distanceToSquared(ref) > despawn2) this.vehicles.despawn(car);
    }

    this.spawnTimer -= dt;
    if (this.spawnTimer <= 0) {
      this.spawnTimer = T.spawnInterval;
      this.frustum.setFromProjectionMatrix(this.projView.multiplyMatrices(this.game.viewCamera.projectionMatrix, this.game.viewCamera.matrixWorldInverse));
      for (let n = 0; n < T.spawnsPerTick && this.cars.length < this.target; n++) {
        if (!this.trySpawn(ref)) break;
      }
    }
    this.game.perf.set('traffic', this.cars.length);
  }

  private trySpawn(ref: V3): boolean {
    const vehicles = this.vehicles as IVehiclesPeer;
    const lp = this.graph.randomLanePointNear(ref, T.spawnMin, T.spawnMax, this.rng, (e) => e.kind !== 'dirt');
    if (!lp) return false;
    const p = lp.point;
    if (this.frustum.containsPoint(p)) return false;
    if (vehicles.query(p, T.spawnClearance, this.scratch).length > 0) return false;
    const id = pickArchetypeFor(this.game, p, this.rng, true);
    const heading = Math.atan2(lp.direction.x, lp.direction.z);
    const driver = new TrafficDriver(this, lp);
    const v = vehicles.spawn(id, { x: p.x, y: p.y + 0.05, z: p.z }, heading, { color: pickColor(id, this.rng), driver }) as Vehicle;
    v.driverTypeId = this.rng.pick(TRAFFIC_DRIVER_TYPES);
    // Start rolling at the road speed so a freshly spawned car never appears parked in a lane.
    const s = Math.min(lp.edge.speedLimit, T.driver.speedCeiling) * 0.8;
    v.body.setLinvel({ x: lp.direction.x * s, y: 0, z: lp.direction.z * s }, true);
    this.cars.push(v);
    return true;
  }

  /** Driver bails: the car brakes to a stop and (with peds loaded) a fleeing ped appears at the door. */
  private bail(car: Vehicle): void {
    remove(this.cars, car);
    const typeId = car.driverTypeId;
    car.setDriver(null);
    car.controls.brake = 1;
    car.driverTypeId = null;
    if (typeId && !car.state.exploded) {
      car.seatDoorPosition(0, this.door);
      this.game.peds?.spawn(typeId, this.door, { heading: car.heading + Math.PI / 2 });
    }
    if (!this.abandoned.includes(car)) this.abandoned.push(car);
  }

  /** The player took this car; drop it from the traffic pool. */
  private release(vehicle: IVehicle): void {
    const car = vehicle as Vehicle;
    if (!(car instanceof Vehicle)) return;
    remove(this.cars, car);
    remove(this.abandoned, car);
    car.driverTypeId = null;
  }

  dispose(): void {
    for (const off of this.offs) off();
    this.offs.length = 0;
    if (this.vehicles) {
      for (let i = this.cars.length - 1; i >= 0; i--) this.vehicles.despawn(this.cars[i]);
      for (let i = this.abandoned.length - 1; i >= 0; i--) this.vehicles.despawn(this.abandoned[i]);
    }
    this.cars.length = 0;
    this.abandoned.length = 0;
  }

  // --- Debug ---------------------------------------------------------------------------

  debugActions = {
    setDensity: (n: unknown): number => {
      this.setDensity(Number(n));
      return this.target;
    },
    count: (): { count: number; target: number; abandoned: number } => ({ count: this.cars.length, target: this.target, abandoned: this.abandoned.length }),
    panic: (): number => {
      if (!this.game.systems.has('player')) return 0;
      const before = this.cars.length;
      this.panicNear(this.game.player.position, 60);
      return before - this.cars.length;
    },
    lights: (): { node: number; green: string; yellow: boolean }[] => {
      if (!this.lights) return [];
      const state = { green: null as 'ns' | 'ew' | null, yellow: false, timeToRed: Infinity };
      return this.lights.lightNodes.slice(0, 8).map((n) => {
        this.lights.stateFor(n.id, n.edges[0], state);
        return { node: n.id, green: state.green ?? 'none', yellow: state.yellow };
      });
    },
    stuck: (): number => this.cars.filter((c) => c.driver instanceof TrafficDriver && c.driver.stuck > 10).length,
    /** Cars idling with a green light and a clear road for more than `seconds`. */
    stalled: (seconds: unknown = 10): number =>
      this.cars.filter((c) => c.driver instanceof TrafficDriver && c.driver.idle > Number(seconds)).length,
  };
}

function remove<T>(list: T[], item: T): void {
  const idx = list.indexOf(item);
  if (idx >= 0) list.splice(idx, 1);
}

const module: SystemModule = {
  id: 'traffic',
  priority: PRIORITY.traffic,
  create: (game) => new TrafficSystem(game),
};

export default module;
