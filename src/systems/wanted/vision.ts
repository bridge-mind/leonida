import { Vector3 } from 'three';
import type { Entity, IPed, V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import { Groups, type RaycastHit } from '../../physics/Physics';
import { WantedConfig } from './config';

const V = WantedConfig.vision;

/**
 * Throttled cop line-of-sight: every `interval` seconds it sweeps up to `raysPerCheck`
 * cops (round-robin over everyone in range) from eye to player chest. Cops sitting in a
 * cruiser look from above the roof so the ray clears their own body. Other units
 * (helicopters, cop brains that already aim) push sightings in via `markSeen`.
 */
export class CopVision {
  private timer = 0;
  private cursor = 0;
  private lastHitAt = -Infinity;
  private readonly cops: IPed[] = [];
  private readonly origin = new Vector3();
  private readonly target = new Vector3();
  private readonly dir = new Vector3();
  private readonly hit: RaycastHit = { entity: null, point: new Vector3(), normal: new Vector3(), toi: 0, collider: null as unknown as RaycastHit['collider'] };

  constructor(private readonly game: Game) {}

  seen(now: number): boolean {
    return now - this.lastHitAt <= V.seenWindow;
  }

  markSeen(now: number): void {
    this.lastHitAt = now;
  }

  reset(): void {
    this.lastHitAt = -Infinity;
    this.cursor = 0;
  }

  update(dt: number, now: number, playerPos: V3): void {
    this.timer += dt;
    if (this.timer < V.interval) return;
    this.timer = 0;
    const peds = this.game.peds;
    if (!peds || !this.game.systems.has('player')) return;
    const player = this.game.player.entity;
    if (!player.isAlive) return;

    const mounted = player.mounted?.vehicle ?? null;
    const targetEntity: Entity = mounted ?? player;
    this.target.set(playerPos.x, playerPos.y + (mounted ? V.vehicleTargetHeight : V.chestHeight), playerPos.z);

    peds.query(playerPos, V.vehicleRange, this.cops, isCop);
    const n = this.cops.length;
    if (n === 0) return;
    const rays = Math.min(V.raysPerCheck, n);
    for (let i = 0; i < rays; i++) {
      const cop = this.cops[(this.cursor + i) % n];
      if (this.canSee(cop, targetEntity)) {
        this.lastHitAt = now;
        break;
      }
    }
    this.cursor = (this.cursor + rays) % n;
  }

  private canSee(cop: IPed, targetEntity: Entity): boolean {
    if (!cop.isAlive || cop.health <= 0) return false;
    const vehicle = cop.vehicle;
    const range = vehicle ? V.vehicleRange : V.footRange;
    if (vehicle) this.origin.set(vehicle.position.x, vehicle.position.y + V.vehicleEyeHeight, vehicle.position.z);
    else this.origin.set(cop.position.x, cop.position.y + V.eyeHeight, cop.position.z);
    this.dir.subVectors(this.target, this.origin);
    const dist = this.dir.length();
    if (dist > range) return false;
    if (dist < V.nearAutoSee) return true;
    this.dir.divideScalar(dist);
    const h = this.game.physics.raycastInto(this.hit, this.origin, this.dir, dist, {
      groups: Groups.STATIC | Groups.VEHICLE,
      exclude: targetEntity,
      solid: false,
    });
    // Leaving the cop's own cruiser is not an obstruction.
    return h === null || h.entity === targetEntity || (vehicle !== null && h.entity === vehicle);
  }
}

function isCop(p: IPed): boolean {
  return p.isCop;
}
