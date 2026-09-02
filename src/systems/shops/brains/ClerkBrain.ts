import type { IPed, PedBrain, PedContext, PedEvent } from '../../../core/entities';
import type { Post } from '../../../world/interiors/StoreKit';
import { asBrainEvent } from '../../peds/events';
import { ShopsConfig } from '../config';
import { shootPlayer } from './fire';

export type ClerkMode = 'idle' | 'handsUp' | 'hostile';

export interface ClerkHooks {
  /** Called once when the clerk reaches under the counter. */
  onDrewShotgun(): void;
}

/**
 * Counter clerk. Stands at his post and watches the player. Two ways to get hurt:
 *  - keep a gun out without covering him for `drawnGraceSeconds` (armed clerks only),
 *  - during a hold-up, let your aim drift off him for longer than `aimGapSeconds`
 *    once `robberyGraceSeconds` have passed.
 * Whether a given clerk is armed is rolled per visit by the shop system (25%).
 */
export class ClerkBrain implements PedBrain {
  mode: ClerkMode = 'idle';
  /** Written each frame by the shop system: player has a weapon out inside this store. */
  playerArmed = false;
  private sinceAimed = Number.POSITIVE_INFINITY;
  private threatTimer = 0;
  private robberyTime = 0;
  private fireTimer = 0;
  private posted = false;

  constructor(
    private readonly post: Post,
    readonly hasShotgun: boolean,
    private readonly hooks: ClerkHooks,
  ) {}

  beginRobbery(ped: IPed): void {
    if (this.mode === 'hostile') return;
    this.mode = 'handsUp';
    this.robberyTime = 0;
    ped.handsUp = true;
    ped.stop();
  }

  update(ped: IPed, ctx: PedContext, dt: number): void {
    const C = ShopsConfig.clerk;
    this.sinceAimed += dt;
    if (!this.posted) {
      ped.state = 'scripted';
      ped.heading = this.post.heading;
      this.posted = true;
    }

    if (this.mode === 'hostile') {
      ped.state = 'fight';
      ped.face(ctx.playerPosition);
      this.fireTimer -= dt;
      if (this.fireTimer <= 0 && ctx.playerDistance < 30) {
        if (shootPlayer(ped, ctx, 'shotgun', C.accuracy)) this.fireTimer = C.fireInterval;
        else this.fireTimer = C.fireInterval * 2;
      }
      return;
    }

    ped.stop();
    if (ctx.playerDistance < C.attentionRadius) ped.face(ctx.playerPosition);
    else ped.heading = this.post.heading;

    if (this.mode === 'handsUp') {
      ped.handsUp = true;
      this.robberyTime += dt;
      if (this.hasShotgun && this.robberyTime > C.robberyGraceSeconds && this.sinceAimed > C.aimGapSeconds) this.drawShotgun(ped);
      return;
    }

    ped.handsUp = false;
    if (this.playerArmed && this.sinceAimed > C.aimGapSeconds) {
      this.threatTimer += dt;
      if (this.threatTimer > C.drawnGraceSeconds && this.hasShotgun) this.drawShotgun(ped);
    } else {
      this.threatTimer = Math.max(0, this.threatTimer - dt);
    }
  }

  onEvent(ped: IPed, event: PedEvent): void {
    const e = asBrainEvent(event);
    if (e.type === 'aimedAt') {
      this.sinceAimed = 0;
      return;
    }
    if (e.type === 'damaged' && this.mode !== 'hostile') {
      // Shot at: an armed clerk shoots back, an unarmed one gives up.
      if (this.hasShotgun) this.drawShotgun(ped);
      else {
        this.mode = 'handsUp';
        ped.handsUp = true;
      }
    }
  }

  private drawShotgun(ped: IPed): void {
    this.mode = 'hostile';
    ped.weaponId = 'shotgun';
    ped.handsUp = false;
    ped.state = 'fight';
    this.fireTimer = 0.4;
    this.hooks.onDrewShotgun();
  }
}
