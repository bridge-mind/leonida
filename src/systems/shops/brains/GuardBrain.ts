import type { IPed, PedBrain, PedContext, PedEvent } from '../../../core/entities';
import type { Post } from '../../../world/interiors/StoreKit';
import { asBrainEvent } from '../../peds/events';
import { ShopsConfig } from '../config';
import { shootPlayer } from './fire';

export interface GuardHooks {
  onWarning(): void;
  onHostile(): void;
}

/**
 * Armed security guard for the jewelry store and the bank. Holds a post, tracks
 * the player, and opens fire shortly after seeing a drawn weapon (or being shot).
 * Once hostile he stays hostile until the player leaves and he is despawned.
 */
export class GuardBrain implements PedBrain {
  /** Written each frame by the shop system: the player has a weapon out in this interior. */
  playerArmed = false;
  private hostile = false;
  private warned = false;
  private warningTimer = 0;
  private fireTimer = 0;
  private posted = false;

  constructor(
    private readonly post: Post,
    private readonly hooks: GuardHooks,
  ) {}

  get isHostile(): boolean {
    return this.hostile;
  }

  update(ped: IPed, ctx: PedContext, dt: number): void {
    const C = ShopsConfig.guard;
    if (!this.posted) {
      ped.state = 'scripted';
      ped.heading = this.post.heading;
      this.posted = true;
    }
    ped.stop();

    if (this.hostile) {
      ped.state = 'fight';
      ped.face(ctx.playerPosition);
      this.fireTimer -= dt;
      if (this.fireTimer <= 0 && ctx.playerDistance < 40) {
        if (shootPlayer(ped, ctx, 'pistol', C.accuracy)) this.fireTimer = C.fireInterval;
        else this.fireTimer = C.fireInterval * 2;
      }
      return;
    }

    if (ctx.playerDistance < C.attentionRadius) ped.face(ctx.playerPosition);
    else ped.heading = this.post.heading;

    if (this.playerArmed) {
      if (!this.warned) {
        this.warned = true;
        this.hooks.onWarning();
      }
      this.warningTimer += dt;
      if (this.warningTimer >= C.warningSeconds) this.goHostile(ped);
    } else {
      this.warningTimer = 0;
    }
  }

  onEvent(ped: IPed, event: PedEvent): void {
    const e = asBrainEvent(event);
    // The ped system broadcasts gunfire as a loud `crimeSeen` (nothing emits `heardGunshot`).
    if (e.type === 'damaged' || e.type === 'heardGunshot' || e.type === 'explosion' || (e.type === 'crimeSeen' && e.loud)) this.goHostile(ped);
  }

  goHostile(ped: IPed): void {
    if (this.hostile) return;
    this.hostile = true;
    ped.weaponId = 'pistol';
    ped.state = 'fight';
    this.fireTimer = 0.2;
    this.hooks.onHostile();
  }
}
