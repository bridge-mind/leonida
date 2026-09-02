import { isDamageable, type DamageInfo, type Entity, type IPed, type IVehicle, type V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { IDamagePeer } from '../../core/peers';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { isDummy } from '../weapons/Dummy';
import type { WeaponSystem } from '../weapons';
import { DamageConfig as C } from './config';
import { explosionFalloff, partMultiplier } from './falloff';

/**
 * Single entry point for all damage (`game.damage.apply`): headshot multiplier, routing to
 * the player or any `IDamageable`, kill detection with attribution, crimes, hit markers and
 * radius explosions with linear falloff.
 */
export class DamageSystem implements GameSystem, IDamagePeer {
  /** Entity ids already reported dead, so `entity:killed` fires once per life. */
  private readonly killed = new Set<number>();

  constructor(private readonly game: Game) {}

  init(): void {}

  // --- IDamagePeer ---------------------------------------------------------------

  /** Applies `info` to `target`; `info.amount` is scaled in place for headshots. Returns false when the target cannot take damage. */
  apply(target: Entity, info: DamageInfo): boolean {
    const aliveBefore = isEntityAlive(target);
    if (aliveBefore) this.killed.delete(target.id);
    info.amount *= partMultiplier(info.part, C.headshotMultiplier, target.kind === 'ped' || isDummy(target));

    if (target.kind === 'player') {
      if (!this.game.systems.has('player')) return false;
      this.game.player.applyDamage(info);
    } else if (isDamageable(target)) {
      target.applyDamage(info);
    } else {
      return false;
    }
    this.game.events.emit('damage:applied', { target, info });

    const killedNow = aliveBefore && !isEntityAlive(target) && !this.killed.has(target.id);
    if (killedNow) {
      this.killed.add(target.id);
      this.game.events.emit('entity:killed', { target, source: info.source ?? null });
      if (info.source === 'player' && target.kind === 'ped') {
        const stats = this.game.state.stats;
        stats.kills = (stats.kills ?? 0) + 1;
        const isCop = (target as IPed).isCop === true;
        this.game.events.emit('crime:committed', {
          type: isCop ? 'kill_cop' : 'kill_ped',
          position: { x: target.position.x, y: target.position.y, z: target.position.z },
          actor: 'player',
          loud: false,
        });
      }
    }
    if (info.source === 'player' && target.kind !== 'player') {
      this.game.events.emit('weapon:hit', { target, killed: killedNow });
    }
    return true;
  }

  explode(position: V3, radius: number, damage: number, impulse: number, source?: Entity | 'player'): void {
    const game = this.game;
    const E = C.explosion;
    const falloff = (dist: number): number => explosionFalloff(dist, radius, E.fullDamageFraction);
    // Explosions are rare and may cascade (a vehicle exploding inside this call), so
    // local arrays are used instead of shared scratch buffers.
    const dir: V3 = { x: 0, y: 0, z: 0 };
    const hit = (target: Entity, dist: number, type: DamageInfo['type'] = 'explosion'): number => {
      const k = falloff(dist);
      if (k <= 0) return 0;
      const len = dist > 1e-3 ? dist : 1;
      dir.x = (target.position.x - position.x) / len;
      dir.y = (target.position.y + 0.5 - position.y) / len;
      dir.z = (target.position.z - position.z) / len;
      this.apply(target, {
        amount: damage * k,
        type,
        point: { x: target.position.x, y: target.position.y + 0.8, z: target.position.z },
        direction: { x: dir.x, y: dir.y, z: dir.z },
        source,
      });
      return k;
    };

    const peds: IPed[] = [];
    game.peds?.query(position, radius, peds);
    for (const ped of peds) {
      const d = distance(position, ped.position);
      if (hit(ped, d) <= 0) continue;
      const push = impulse * (0.5 + 0.5 * falloff(d));
      ped.ragdoll({ x: dir.x * push * E.pedImpulseForward, y: push * E.pedImpulseUp, z: dir.z * push * E.pedImpulseForward });
    }

    const vehicles: IVehicle[] = [];
    game.vehicles?.query(position, radius, vehicles);
    for (const v of vehicles) hit(v, distance(position, v.position));

    const weapons = game.systems.get<WeaponSystem>('weapons');
    if (weapons) for (const dummy of weapons.dummies) hit(dummy, distance(position, dummy.position));

    let playerDistance = Infinity;
    if (game.systems.has('player')) {
      const player = game.player.entity;
      playerDistance = distance(position, player.position);
      if (player.isAlive) hit(player, playerDistance);
    }

    game.particles?.emit('explosion', position, { scale: radius / E.particleScaleDivisor });
    game.audio.play('explosion.large', { position });
    if (game.systems.has('camera') && playerDistance < E.shakeRange) {
      const k = 1 - playerDistance / E.shakeRange;
      game.camera.shake(E.shakeStrength * k, E.shakeSeconds * (0.5 + 0.5 * k));
    }
    if (source === 'player') {
      game.events.emit('crime:committed', { type: 'explosion', position: { x: position.x, y: position.y, z: position.z }, actor: 'player', loud: true });
    }
  }

  debugActions = {
    /** `damage.explode:<x>:<z>:<radius>:<damage>[:<impulse>]` at ground level, attributed to the player. */
    explode: (x: unknown, z: unknown, radius: unknown, damage: unknown, impulse?: unknown): void => {
      const px = Number(x);
      const pz = Number(z);
      const y = (this.game.physics.groundHeightAt(px, pz) ?? 0) + 0.5;
      this.explode({ x: px, y, z: pz }, Number(radius ?? 6), Number(damage ?? 100), Number(impulse ?? 20000), 'player');
    },
    /** Damages the player (`damage.player:<amount>`). */
    player: (amount: unknown): number => {
      if (!this.game.systems.has('player')) return 0;
      this.apply(this.game.player.entity, { amount: Number(amount), type: 'bullet' });
      return this.game.player.health;
    },
  };
}

/** Alive unless the entity says otherwise or exposes a non-positive `health`. */
export function isEntityAlive(e: Entity): boolean {
  if (!e.isAlive) return false;
  const health = (e as { health?: unknown }).health;
  return !(typeof health === 'number' && health <= 0);
}

function distance(a: V3, b: V3): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

const module: SystemModule = {
  id: 'damage',
  priority: PRIORITY.damage,
  create: (game) => new DamageSystem(game),
};

export default module;
