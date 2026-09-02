import { Vector3 } from 'three';
import { isDamageable, type Entity, type IVehicle } from '../../core/entities';
import type { Game } from '../../core/Game';
import { Vehicle } from '../../entities/Vehicle';
import { VehicleConfig } from './config';

const D = VehicleConfig.damage;
const tmp = new Vector3();
const tmp2 = new Vector3();
const nearby: IVehicle[] = [];

/**
 * Damage tiers, smoke, fire and the explosion. Kept out of the entity so the
 * Vehicle file stays focused on physics and the IVehicle contract.
 */

/** Per fixed step: smoke below 300 hp, fire below 0, explosion after the fire timer. */
export function tickVehicleFx(game: Game, v: Vehicle, dt: number): void {
  const s = v.state;
  if (s.exploded) {
    v.wreckTimer += dt;
    return;
  }
  if (s.onFire) {
    v.fireFxTimer -= dt;
    if (v.fireFxTimer <= 0) {
      v.fireFxTimer = D.fireInterval;
      game.particles?.emit('fire', v.hoodPosition(tmp), { scale: 1.2 });
    }
    burnOccupants(game, v, dt);
    v.fireTimer -= dt;
    if (v.fireTimer <= 0) v.explode();
    return;
  }
  if (s.health < D.smokeHealth) {
    v.smokeFxTimer -= dt;
    if (v.smokeFxTimer <= 0) {
      v.smokeFxTimer = D.smokeInterval;
      game.particles?.emit('smoke', v.hoodPosition(tmp));
    }
  }
}

function burnOccupants(game: Game, v: Vehicle, dt: number): void {
  const amount = D.fireDps * dt;
  const occupants = v.state.occupants;
  for (let i = 0; i < occupants.length; i++) {
    const o = occupants[i];
    if (o.kind === 'player' && game.systems.has('player')) game.player.applyDamage({ amount, type: 'fire', source: v });
    else if (isDamageable(o) && !game.damage?.apply(o, { amount, type: 'fire', source: v })) o.applyDamage({ amount, type: 'fire', source: v });
  }
}

/** Fireball, radius damage, occupant ejection, wreck state. */
export function explodeVehicle(game: Game, v: Vehicle): void {
  const s = v.state;
  s.exploded = true;
  s.onFire = false;
  s.wrecked = true;
  s.health = Math.min(0, s.health);
  s.lightsOn = false;
  s.sirenOn = false;
  v.setDriver(null);
  const pos = tmp.copy(v.position);
  pos.y += 0.6;
  const byPlayer = v.lastDamageByPlayer;
  const source: Entity | 'player' = byPlayer ? 'player' : v;

  // Eject the occupants before the radius damage so the player takes it on foot.
  const occupants = s.occupants.slice();
  for (const o of occupants) {
    if (o.kind === 'player' && game.systems.has('player')) {
      const door = v.seatDoorPosition(0, tmp2);
      v.exit(o);
      game.player.dismount(door);
      if (!game.damage) game.player.applyDamage({ amount: D.explosionDamage, type: 'explosion', source });
    } else {
      v.exit(o);
      // Via the damage system so occupant deaths emit `entity:killed` once (it dedupes by id).
      if (isDamageable(o) && !game.damage?.apply(o, { amount: 1000, type: 'explosion', source })) o.applyDamage({ amount: 1000, type: 'explosion', source });
    }
  }

  const damage = game.damage;
  if (damage) {
    damage.explode(pos, D.explosionRadius, D.explosionDamage, D.explosionImpulse, source);
  } else {
    applyFallbackRadiusDamage(game, v, pos, source);
  }
  game.particles?.emit('explosion', pos, { scale: 1.5 });
  game.audio.play('vehicle.explosion', { position: pos });
  if (game.systems.has('camera')) game.camera.shake(0.6, 0.7);
  // A hop sells the blast even with pitch/roll locked.
  v.body.wakeUp();
  v.body.applyImpulse({ x: 0, y: v.archetype.mass * 3.5, z: 0 }, true);
  v.wreckTimer = 0;
  game.events.emit('vehicle:exploded', { vehicle: v, position: { x: pos.x, y: pos.y, z: pos.z } });
  if (byPlayer) game.events.emit('crime:committed', { type: 'explosion', position: { x: pos.x, y: pos.y, z: pos.z }, actor: 'player', loud: true });
}

function applyFallbackRadiusDamage(game: Game, v: Vehicle, pos: Vector3, source: Entity | 'player'): void {
  const r = D.explosionRadius;
  if (game.systems.has('player') && !game.player.isMounted) {
    const p = game.player.position;
    if (p.distanceTo(pos) <= r) game.player.applyDamage({ amount: D.explosionDamage, type: 'explosion', source });
  }
  const vehicles = game.vehicles;
  if (!vehicles) return;
  vehicles.query(pos, r, nearby);
  for (let i = 0; i < nearby.length; i++) {
    const other = nearby[i];
    if (other === v) continue;
    other.applyDamage({ amount: D.explosionDamage, type: 'explosion', source, point: pos });
    if (other instanceof Vehicle) {
      tmp2.subVectors(other.position, pos);
      tmp2.y = 0.3;
      const len = Math.max(0.5, tmp2.length());
      tmp2.multiplyScalar((D.explosionImpulse * 0.6) / len);
      other.body.wakeUp();
      other.body.applyImpulse(tmp2, true);
    }
  }
}

/**
 * Called from `physics:contactStarted`: damage from this vehicle's own velocity change
 * across the step (impulse = mass · Δv), ignoring parking bumps.
 */
export function handleVehicleContact(game: Game, v: Vehicle, other: Entity | null): void {
  if (!v.isAlive) return;
  const lv = v.body.linvel();
  const dvx = lv.x - v.prevLinvel.x;
  const dvy = lv.y - v.prevLinvel.y;
  const dvz = lv.z - v.prevLinvel.z;
  const deltaV = Math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
  if (deltaV < D.minDeltaV) return;
  const impulse = deltaV * v.archetype.mass;
  game.events.emit('vehicle:collided', { vehicle: v, other, impulse });
  const otherVehicle = other instanceof Vehicle ? other : null;
  const source: Entity | 'player' | undefined = otherVehicle?.playerDriven ? 'player' : (other ?? undefined);
  v.applyDamage({ amount: (deltaV - D.minDeltaV) * D.damagePerDeltaV * (v.archetype.mass / 1200), type: 'vehicle', source });
  const sound = deltaV >= D.crashHeavy ? 'vehicle.crash.heavy' : deltaV >= D.crashMedium ? 'vehicle.crash.medium' : 'vehicle.crash.light';
  game.audio.play(sound, { position: v.position, volume: Math.min(1, 0.4 + deltaV / 20) });
  if (v.playerDriven && game.systems.has('camera')) {
    game.camera.shake(Math.min(D.maxCameraShake, deltaV * D.cameraShakePerDeltaV), 0.35);
  }
  // Consume the change so a multi-step crash is not counted twice.
  v.prevLinvel.set(lv.x, lv.y, lv.z);
}

/** Body colour multiplier for the current health tier. */
export function damageTier(v: Vehicle): number {
  if (v.state.wrecked) return 3;
  const h = v.state.health;
  return h >= 700 ? 0 : h >= 300 ? 1 : 2;
}
