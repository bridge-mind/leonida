import { Vector3 } from 'three';
import { isDamageable, type DamageInfo, type Entity, type IPed, type V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import { Groups, type RaycastHit } from '../../physics/Physics';
import { WeaponsConfig } from './config';
import type { Hitscan } from './Hitscan';

const M = WeaponsConfig.melee;

/**
 * Unarmed 3-hit combo. Targets come from the ped spatial query (facing test) or, without a
 * peds system, from a short chest-height raycast so dummies and vehicles can be punched.
 */
export class Melee {
  combo = 0;
  private cooldown = 0;
  private sinceHit = Infinity;
  private readonly peds: IPed[] = [];
  private readonly forward = new Vector3();
  private readonly toTarget = new Vector3();
  private readonly origin = new Vector3();
  private readonly hit: RaycastHit = {
    entity: null,
    point: new Vector3(),
    normal: new Vector3(),
    toi: 0,
    collider: null as unknown as RaycastHit['collider'],
  };

  constructor(
    private readonly game: Game,
    private readonly hitscan: Hitscan,
  ) {}

  update(dt: number): void {
    this.cooldown -= dt;
    this.sinceHit += dt;
    if (this.sinceHit > M.comboReset) this.combo = 0;
  }

  /** Attempts a punch; returns false while the previous hit is still recovering. */
  punch(): boolean {
    if (this.cooldown > 0) return false;
    const player = this.game.player;
    const pos = player.position;
    this.cooldown = M.hitInterval;
    this.sinceHit = 0;
    this.combo = (this.combo % 3) + 1;
    // Rig heading 0 faces +Z.
    this.forward.set(Math.sin(player.heading), 0, Math.cos(player.heading));
    const target = this.findTarget(pos);
    this.game.audio.play('weapon.melee.swing', { position: pos });
    if (target) {
      const info: DamageInfo = {
        amount: M.damage,
        type: 'melee',
        point: { x: target.position.x, y: target.position.y + 1.2, z: target.position.z },
        direction: { x: this.forward.x, y: 0, z: this.forward.z },
        part: 'body',
        source: 'player',
      };
      this.hitscan.applyDamage(target, info);
      this.game.audio.play('weapon.melee.hit', { position: target.position });
      this.game.particles?.emit(target.kind === 'ped' ? 'blood' : 'dust', info.point as V3, { direction: this.forward, scale: 0.6 });
      const ragdoll = (target as { ragdoll?: (impulse: V3) => void }).ragdoll;
      if (this.combo === 3 && typeof ragdoll === 'function' && target.kind === 'ped') {
        ragdoll.call(target, { x: this.forward.x * M.knockdownForward, y: M.knockdownUp, z: this.forward.z * M.knockdownForward });
        this.game.audio.play('weapon.melee.knockdown', { position: target.position });
      }
    }
    if (this.game.systems.has('camera')) {
      this.game.camera.shake(0.015, 0.1);
      this.game.camera.fovKick(this.combo === 3 ? 3 : 1.5);
    }
    this.game.events.emit('weapon:meleeSwung', { combo: this.combo, hit: target });
    return true;
  }

  private findTarget(pos: V3): Entity | null {
    const peds = this.game.peds;
    if (peds) {
      this.peds.length = 0;
      peds.query(pos, M.queryRadius, this.peds, (p) => p.isAlive && p.state !== 'dead');
      let best: IPed | null = null;
      let bestDist = Infinity;
      for (const p of this.peds) {
        this.toTarget.set(p.position.x - pos.x, 0, p.position.z - pos.z);
        const d = this.toTarget.length();
        if (d > M.reach + 0.4 || d < 1e-3) continue;
        this.toTarget.divideScalar(d);
        if (this.toTarget.dot(this.forward) < M.facingDot) continue;
        if (d < bestDist) {
          bestDist = d;
          best = p;
        }
      }
      if (best) return best;
    }
    this.origin.set(pos.x, pos.y + 1.2, pos.z);
    const h = this.game.physics.raycastInto(this.hit, this.origin, this.forward, M.reach + 0.35, {
      groups: Groups.PED | Groups.VEHICLE,
      exclude: this.game.player.entity,
    });
    return h && h.entity && isDamageable(h.entity) ? h.entity : null;
  }
}
