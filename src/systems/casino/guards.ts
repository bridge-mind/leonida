import { Vector3 } from 'three';
import type { IPed, PedBrain, PedContext, V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { WeaponId } from '../../core/peers';
import { CasinoConfig } from './config';

/**
 * Door security. Idle guards watch the player; once alerted they draw pistols and
 * fire through `weapons.npcFire` every `fireInterval` seconds. Everything degrades to
 * a no-op when the peds (or weapons) system is not loaded.
 */
class GuardBrain implements PedBrain {
  hostile = false;
  private cooldown = 0;
  private readonly eye = new Vector3();
  private readonly dir = new Vector3();

  constructor(private readonly game: Game) {}

  update(ped: IPed, ctx: PedContext, dt: number): void {
    ped.face(ctx.playerPosition);
    if (!this.hostile) return;
    this.cooldown -= dt;
    if (this.cooldown > 0) return;
    this.cooldown = CasinoConfig.guards.fireInterval;
    const weapons = this.game.weapons;
    if (!weapons) return;
    const p = ped.position;
    const t = ctx.playerPosition;
    this.eye.set(p.x, p.y + CasinoConfig.guards.eyeHeight, p.z);
    this.dir.set(t.x - this.eye.x, t.y + 1.2 - this.eye.y, t.z - this.eye.z).normalize();
    weapons.npcFire(ped, CasinoConfig.guards.weaponId as WeaponId, this.eye, this.dir, CasinoConfig.guards.accuracy);
  }
}

export class CasinoGuards {
  private readonly guards: IPed[] = [];
  private readonly brains: GuardBrain[] = [];
  hostile = false;

  constructor(
    private readonly game: Game,
    private readonly origin: V3,
  ) {}

  get count(): number {
    return this.guards.length;
  }

  spawn(): void {
    const peds = this.game.peds;
    if (!peds || this.guards.length > 0) return;
    const typeId = peds.typeIds.includes(CasinoConfig.guards.typeId) ? CasinoConfig.guards.typeId : peds.typeIds[0];
    if (!typeId) return;
    for (const post of CasinoConfig.guards.posts) {
      const brain = new GuardBrain(this.game);
      brain.hostile = this.hostile;
      const ped = peds.spawn(
        typeId,
        { x: post.x + this.origin.x, y: post.y + this.origin.y, z: post.z + this.origin.z },
        { brain, heading: post.heading, weaponId: this.hostile ? (CasinoConfig.guards.weaponId as WeaponId) : undefined },
      );
      ped.state = 'scripted';
      this.guards.push(ped);
      this.brains.push(brain);
    }
  }

  /** Weapons out: every guard turns hostile. Safe to call with no guards spawned. */
  alert(): void {
    this.hostile = true;
    for (let i = 0; i < this.guards.length; i++) {
      this.guards[i].weaponId = CasinoConfig.guards.weaponId;
      this.brains[i].hostile = true;
    }
    if (this.guards.length > 0) this.game.events.emit('casino:guardsAlerted', { count: this.guards.length });
  }

  despawn(): void {
    const peds = this.game.peds;
    for (const g of this.guards) if (peds && g.isAlive) peds.despawn(g);
    this.guards.length = 0;
    this.brains.length = 0;
  }

  /** Guards stand down for the next visit (after the ban expires). */
  reset(): void {
    this.hostile = false;
  }
}
