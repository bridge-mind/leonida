import { Frustum, Matrix4, Vector3 } from 'three';
import type { IPed, IVehicle, V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { WeaponId } from '../../core/peers';
import type { RoadGraph } from '../../world/RoadGraph';
import { PoliceConfig } from './config';
import type { PoliceContext } from './context';
import { CopBrain } from './CopBrain';
import { PoliceDriver } from './PoliceDriver';
import { SwatBrain } from './SwatBrain';

const S = PoliceConfig.dispatch;
const tmp = new Vector3();
const seat = new Vector3();

export interface CruiserUnit {
  kind: 'police' | 'swat';
  vehicle: IVehicle;
  driver: PoliceDriver;
  crew: IPed[];
  /** The crew has left the car (player on foot nearby or the car was wrecked). */
  bailed: boolean;
}

export interface SpawnPoint {
  position: Vector3;
  heading: number;
}

/** Bookkeeping the dispatcher needs from the police system. */
export interface DispatchHost {
  readonly ctx: PoliceContext;
  readonly graph: RoadGraph;
  registerCop(ped: IPed, brain: CopBrain): void;
}

/**
 * Spawns police units: finds off-screen lane points 150–250 m from the player, creates
 * cruisers / SWAT vans with a PoliceDriver and a crew already seated, and foot cops.
 */
export class Dispatcher {
  private readonly frustum = new Frustum();
  private readonly projView = new Matrix4();
  private readonly scratch: IVehicle[] = [];
  private readonly point: SpawnPoint = { position: new Vector3(), heading: 0 };

  constructor(private readonly host: DispatchHost) {}

  get game(): Game {
    return this.host.ctx.game;
  }

  refreshFrustum(): void {
    const cam = this.host.ctx.game.viewCamera;
    this.frustum.setFromProjectionMatrix(this.projView.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse));
  }

  inView(p: V3): boolean {
    tmp.set(p.x, p.y + 1, p.z);
    return this.frustum.containsPoint(tmp);
  }

  /** Off-screen lane point in the spawn ring, clear of other cars; null when none found. */
  findSpawnPoint(ref: V3): SpawnPoint | null {
    const ctx = this.host.ctx;
    const vehicles = ctx.game.vehicles;
    for (let i = 0; i < S.spawnTries; i++) {
      const lp = this.host.graph.randomLanePointNear(ref, S.spawnMin, S.spawnMax, ctx.rng, (e) => e.kind !== 'dirt');
      if (!lp) return null;
      if (this.inView(lp.point)) continue;
      if (vehicles && vehicles.query(lp.point, S.spawnClearance, this.scratch).length > 0) continue;
      this.point.position.set(lp.point.x, lp.point.y + 0.05, lp.point.z);
      this.point.heading = Math.atan2(lp.direction.x, lp.direction.z);
      return this.point;
    }
    return null;
  }

  spawnCruiser(kind: 'police' | 'swat', at: SpawnPoint, crewSize: number, weapon: WeaponId): CruiserUnit | null {
    const ctx = this.host.ctx;
    const vehicles = ctx.game.vehicles;
    if (!vehicles) return null;
    const driver = new PoliceDriver(ctx, this.host.graph);
    let vehicle: IVehicle;
    try {
      vehicle = vehicles.spawn(kind, at.position, at.heading, { driver });
    } catch {
      // Renderer slots exhausted: try again next dispatch tick.
      return null;
    }
    vehicle.setSiren(true);
    const unit: CruiserUnit = { kind, vehicle, driver, crew: [], bailed: false };
    const peds = ctx.game.peds;
    if (peds) {
      for (let i = 0; i < crewSize; i++) {
        vehicle.seatDoorPosition(i, seat);
        const brain = kind === 'swat' ? new SwatBrain(ctx, weapon) : new CopBrain(ctx, weapon);
        const ped = peds.spawn(kind === 'swat' ? 'swat' : 'cop', seat, { brain, heading: at.heading, weaponId: weapon });
        if (!ped.enterVehicle(vehicle, i)) {
          // No free seat (archetype smaller than the crew): stand this one down.
          peds.despawn(ped);
          continue;
        }
        ped.state = 'scripted';
        this.host.registerCop(ped, brain);
        unit.crew.push(ped);
      }
    }
    ctx.game.audio.play('police.dispatch', { bus: 'ui', volume: 0.6 });
    return unit;
  }

  spawnParkedCruiser(position: V3, heading: number): IVehicle | null {
    const vehicles = this.host.ctx.game.vehicles;
    if (!vehicles) return null;
    try {
      const v = vehicles.spawn('police', position, heading, { parked: true });
      v.setSiren(true);
      return v;
    } catch {
      return null;
    }
  }

  spawnFootCop(typeId: 'cop' | 'swat', position: V3, heading: number, weapon: WeaponId): IPed | null {
    const ctx = this.host.ctx;
    const peds = ctx.game.peds;
    if (!peds) return null;
    const brain = typeId === 'swat' ? new SwatBrain(ctx, weapon) : new CopBrain(ctx, weapon);
    const ped = peds.spawn(typeId, position, { brain, heading, weaponId: weapon });
    ped.state = 'scripted';
    this.host.registerCop(ped, brain);
    return ped;
  }
}
