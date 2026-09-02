import type { RigidBody } from '@dimforge/rapier3d-compat';
import { BoxGeometry, Mesh, MeshStandardMaterial, Vector3 } from 'three';
import type { IPed, IVehicle, V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { WeaponId } from '../../core/peers';
import type { RoadGraph } from '../../world/RoadGraph';
import { PoliceConfig } from './config';

const R = PoliceConfig.roadblock;
const spikeGeo = new BoxGeometry(R.spikeHalfLength * 2, 0.08, 0.6);
const spikeMat = new MeshStandardMaterial({ color: 0x3a3a3a, roughness: 0.8, metalness: 0.4 });
const tmp = new Vector3();

/** What a roadblock needs from the police system to place units. */
export interface RoadblockHost {
  readonly game: Game;
  spawnParkedCruiser(position: V3, heading: number): IVehicle | null;
  spawnFootCop(typeId: 'cop' | 'swat', position: V3, heading: number, weapon: WeaponId): IPed | null;
}

export interface RoadblockPlacement {
  center: Vector3;
  /** Unit direction of the road at the block. */
  roadDir: Vector3;
}

/**
 * Picks a lane point 120–200 m ahead of the player's travel direction. Returns null
 * when no road lies there (open ground, water).
 */
export function planRoadblock(graph: RoadGraph, playerPos: V3, heading: number, distance: number, out: RoadblockPlacement): RoadblockPlacement | null {
  tmp.set(playerPos.x + Math.sin(heading) * distance, playerPos.y, playerPos.z + Math.cos(heading) * distance);
  const lp = graph.nearestLanePoint(tmp);
  if (!lp) return null;
  // Too far from the probe means the road bends away; a block there would be off-route.
  if (lp.point.distanceToSquared(tmp) > 60 * 60) return null;
  // Lane points sit in one lane; shift to the road center so cruisers straddle both directions.
  const right = tmp.set(lp.direction.z, 0, -lp.direction.x);
  const laneCenterOffset = lp.edge.width / 4;
  out.center.copy(lp.point).addScaledVector(right, -laneCenterOffset);
  out.roadDir.copy(lp.direction).setY(0).normalize();
  return out;
}

/**
 * Two cruisers parked nose-to-nose across the road with foot cops behind them and,
 * at level 5+, a spike strip on the approach side. Cruisers stay until the block is
 * torn down (level cleared or the player drove far past it).
 */
export class Roadblock {
  readonly center = new Vector3();
  readonly vehicles: IVehicle[] = [];
  readonly cops: IPed[] = [];
  private readonly spike: Mesh | null = null;
  private readonly spikeCenter = new Vector3();
  private readonly spikeDir = new Vector3();
  private spikeCooldown = 0;

  constructor(
    private readonly host: RoadblockHost,
    placement: RoadblockPlacement,
    playerPos: V3,
    cops: number,
    spikes: boolean,
    weapon: WeaponId,
  ) {
    const game = host.game;
    this.center.copy(placement.center);
    const road = placement.roadDir;
    // Cruisers face across the road, offset left/right of the center line.
    const rightX = road.z;
    const rightZ = -road.x;
    // Heading whose forward (sin h, cos h) is the road's right vector.
    const across = Math.atan2(rightX, rightZ);
    for (const side of [-1, 1]) {
      const x = this.center.x + rightX * R.lateralSpacing * side;
      const z = this.center.z + rightZ * R.lateralSpacing * side;
      const y = (game.physics.groundHeightAt(x, z) ?? this.center.y) + 0.05;
      const v = host.spawnParkedCruiser({ x, y, z }, across);
      if (v) this.vehicles.push(v);
    }
    // Cops stand on the far side from the player, behind the cars.
    const toPlayerX = playerPos.x - this.center.x;
    const toPlayerZ = playerPos.z - this.center.z;
    const facing = toPlayerX * road.x + toPlayerZ * road.z > 0 ? 1 : -1;
    const behindX = -road.x * facing;
    const behindZ = -road.z * facing;
    for (let i = 0; i < cops; i++) {
      const lateral = (i - (cops - 1) / 2) * 2.4;
      const x = this.center.x + behindX * 4 + rightX * lateral;
      const z = this.center.z + behindZ * 4 + rightZ * lateral;
      const y = (game.physics.groundHeightAt(x, z) ?? this.center.y) + 0.05;
      const ped = host.spawnFootCop('cop', { x, y, z }, Math.atan2(-behindX, -behindZ), weapon);
      if (ped) this.cops.push(ped);
    }
    if (spikes) {
      const strip = new Mesh(spikeGeo, spikeMat);
      const sx = this.center.x - behindX * 7;
      const sz = this.center.z - behindZ * 7;
      const sy = (game.physics.groundHeightAt(sx, sz) ?? this.center.y) + 0.04;
      strip.position.set(sx, sy, sz);
      // BoxGeometry's long axis is local +X; rotation.y maps it to (cos θ, 0, -sin θ).
      strip.rotation.y = Math.atan2(-rightZ, rightX);
      game.worldRoot.add(strip);
      this.spike = strip;
      this.spikeCenter.set(sx, sy, sz);
      this.spikeDir.set(rightX, 0, rightZ);
      game.audio.play('police.spikes.deploy', { position: this.spikeCenter });
    }
    game.audio.play('police.roadblock', { position: this.center });
  }

  /** Spike strip check against the player's car (called every fixed step while the block exists). */
  update(dt: number, playerCar: IVehicle | null): void {
    if (!this.spike) return;
    this.spikeCooldown = Math.max(0, this.spikeCooldown - dt);
    if (!playerCar || !playerCar.isAlive || playerCar.state.speedMs < R.minSpeedForSpike || this.spikeCooldown > 0) return;
    const dx = playerCar.position.x - this.spikeCenter.x;
    const dz = playerCar.position.z - this.spikeCenter.z;
    const along = dx * this.spikeDir.x + dz * this.spikeDir.z;
    const perp = dx * -this.spikeDir.z + dz * this.spikeDir.x;
    if (Math.abs(along) > R.spikeHalfLength || Math.abs(perp) > 1.0) return;
    this.spikeCooldown = R.spikeCooldown;
    const game = this.host.game;
    // Vehicles expose no tyre state, so a spike is modelled as damage plus a sideways shove.
    playerCar.applyDamage({ amount: R.spikeDamage, type: 'vehicle', point: this.spikeCenter });
    const body = (playerCar as { body?: RigidBody }).body;
    if (body) {
      const side = perp >= 0 ? 1 : -1;
      const m = body.mass();
      body.applyImpulse({ x: -this.spikeDir.z * side * R.spikeImpulse * m, y: 0, z: this.spikeDir.x * side * R.spikeImpulse * m }, true);
    }
    game.particles?.emit('tireSmoke', playerCar.position, { scale: 1.5 });
    game.audio.play('vehicle.crash.medium', { position: playerCar.position });
  }

  /** Player passed the block by a wide margin: it no longer matters. */
  isStale(playerPos: V3): boolean {
    const dx = playerPos.x - this.center.x;
    const dz = playerPos.z - this.center.z;
    return dx * dx + dz * dz > R.staleDistance * R.staleDistance;
  }

  dispose(): void {
    const game = this.host.game;
    const vehicles = game.vehicles;
    for (const v of this.vehicles) {
      if (v.isAlive && !(v.driverEntity && v.driverEntity.kind === 'player')) vehicles?.despawn(v);
    }
    this.vehicles.length = 0;
    if (this.spike) game.worldRoot.remove(this.spike);
  }
}
