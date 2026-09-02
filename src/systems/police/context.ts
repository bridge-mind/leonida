import { Vector3 } from 'three';
import type { Entity, IPed, IVehicle, V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { Random } from '../../core/Random';
import type { DispatchTier } from '../../data/police';
import { Groups, type RaycastHit } from '../../physics/Physics';
import '../wanted/events';

/** What drivers, brains and the helicopter need from the police system. */
export interface PoliceContext {
  readonly game: Game;
  readonly level: number;
  readonly heat: number;
  readonly tier: DispatchTier;
  /** True while the player is dead or being arrested: units hold. */
  readonly holdFire: boolean;
  readonly sight: SightLine;
  readonly rng: Random;
  playerPosition(): V3;
  playerVehicle(): IVehicle | null;
  playerAlive(): boolean;
  playerSpeed(): number;
  playerArmed(): boolean;
  /** Called by any unit that has line of sight right now. */
  reportSighting(by: 'cop' | 'cruiser' | 'helicopter'): void;
  /** A cop fired at the player (UI hit direction / audio hooks listen to `ped:attackedPlayer`). */
  copFired(ped: IPed): void;
}

/**
 * Shared line-of-sight raycaster (preallocated hit). Callers throttle themselves;
 * the ray ignores the player's own body/vehicle so hitting the target reads as clear.
 */
export class SightLine {
  private readonly dir = new Vector3();
  private readonly target = new Vector3();
  private readonly hit: RaycastHit = { entity: null, point: new Vector3(), normal: new Vector3(), toi: 0, collider: null as unknown as RaycastHit['collider'] };

  constructor(private readonly game: Game) {}

  /**
   * True when nothing static or vehicular blocks `origin` → player (chest height), within
   * `range`. `ignore` is the observer's own vehicle (a cruiser roof does not block its crew).
   */
  toPlayer(origin: V3, range: number, ignore: Entity | null = null): boolean {
    const game = this.game;
    if (!game.systems.has('player')) return false;
    const player = game.player.entity;
    const mounted = player.mounted?.vehicle ?? null;
    const targetEntity: Entity = mounted ?? player;
    const p = player.position;
    this.target.set(p.x, p.y + (mounted ? 1.0 : 1.2), p.z);
    this.dir.set(this.target.x - origin.x, this.target.y - origin.y, this.target.z - origin.z);
    const dist = this.dir.length();
    if (dist > range) return false;
    if (dist < 1e-3) return true;
    this.dir.divideScalar(dist);
    const h = game.physics.raycastInto(this.hit, origin, this.dir, dist, {
      groups: Groups.STATIC | Groups.VEHICLE,
      exclude: targetEntity,
      solid: false,
    });
    return h === null || h.entity === targetEntity || (ignore !== null && h.entity === ignore);
  }
}

/** Unit direction from `from` to the player's chest, written into `out`. */
export function aimAtPlayer(ctx: PoliceContext, from: V3, out: Vector3): Vector3 {
  const p = ctx.playerPosition();
  const inCar = ctx.playerVehicle() !== null;
  out.set(p.x - from.x, p.y + (inCar ? 1.0 : 1.2) - from.y, p.z - from.z);
  const len = out.length();
  return len > 1e-6 ? out.divideScalar(len) : out.set(0, 0, 1);
}
