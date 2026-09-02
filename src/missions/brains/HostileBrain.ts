import { Vector3 } from 'three';
import type { IPed, PedBrain, PedContext } from '../../core/entities';
import type { WeaponId } from '../../core/peers';

const tmp = new Vector3();
const tmp2 = new Vector3();

export interface HostileOptions {
  weapon: WeaponId;
  /** Shots per second. */
  fireRate: number;
  accuracy: number;
  /** Keeps at least this far from the player; closes in when farther than `maxRange`. */
  minRange: number;
  maxRange: number;
  runSpeed: number;
  /** Peds farther than this from the player just hold position (ambush spawned around the club). */
  engageRange: number;
}

export const HOSTILE_DEFAULTS: HostileOptions = { weapon: 'pistol', fireRate: 1.2, accuracy: 0.45, minRange: 8, maxRange: 18, runSpeed: 5.5, engageRange: 90 };

/**
 * Mission enemy: knows about the player from the start, closes to gun range and fires
 * through `weapons.npcFire`. Stays in the `scripted` state so the ambient spawner keeps it.
 */
export class HostileBrain implements PedBrain {
  private fireTimer: number;
  private readonly opts: HostileOptions;

  constructor(opts: Partial<HostileOptions> = {}) {
    this.opts = { ...HOSTILE_DEFAULTS, ...opts };
    this.fireTimer = 0.4 + Math.random() * 0.8;
  }

  update(ped: IPed, ctx: PedContext, dt: number): void {
    ped.state = 'scripted';
    const game = ctx.game;
    const player = game.systems.has('player') ? game.player : null;
    if (!player || !player.isAlive) {
      ped.stop();
      return;
    }
    const target = ctx.playerPosition;
    const dist = ctx.playerDistance;
    const o = this.opts;
    if (dist > o.engageRange) {
      ped.stop();
      return;
    }
    if (dist > o.maxRange) ped.moveToward(target, o.runSpeed);
    else if (dist < o.minRange) {
      tmp.copy(ped.position).sub(target).setY(0).normalize().multiplyScalar(4).add(ped.position);
      ped.moveToward(tmp, o.runSpeed * 0.6);
    } else ped.stop();
    ped.face(target);
    this.fireTimer -= dt;
    if (this.fireTimer > 0 || dist > o.maxRange * 1.5) return;
    this.fireTimer = 1 / o.fireRate;
    tmp.set(ped.position.x, ped.position.y + 1.6, ped.position.z);
    tmp2.set(target.x, target.y + 1.2, target.z).sub(tmp).normalize();
    game.weapons?.npcFire(ped, o.weapon, tmp, tmp2, o.accuracy);
    game.events.emit('ped:attackedPlayer', { ped, melee: false });
  }
}
