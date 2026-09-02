import { Vector3 } from 'three';
import type { IPed, PedBrain, PedContext, PedEvent, V3 } from '../../core/entities';
import type { WeaponId } from '../../core/peers';
import { COP_FIRE_INTERVAL } from '../../data/police';
import { asBrainEvent } from '../peds/events';
import { PoliceConfig } from './config';
import { aimAtPlayer, type PoliceContext } from './context';

const K = PoliceConfig.cop;
const eye = new Vector3();
const dir = new Vector3();
const tmp = new Vector3();

export type CopMode = 'ride' | 'approach' | 'arrest' | 'combat' | 'standDown';

/**
 * Foot cop. Levels 1–2 against an unarmed, non-fleeing player: walk up and arrest
 * (the system counts the contact time and busts). Otherwise: keep a 12–20 m standoff,
 * strafe, and fire through `weapons.npcFire` whenever there is line of sight. Any
 * damage taken makes the cop lethal for good.
 */
export class CopBrain implements PedBrain {
  mode: CopMode = 'ride';
  provoked = false;
  /** True when this cop is inside arrest range with line of sight (read by the system). */
  arrestContact = false;
  private fireTimer: number = K.firstShotDelay;
  private losTimer = 0;
  private hasLos = false;
  private strafeDir = 1;
  private strafeTimer = 0;
  private readonly runSpeed: number;
  private readonly walkSpeed: number;

  constructor(
    protected readonly ctx: PoliceContext,
    readonly weapon: WeaponId,
    protected readonly swat = false,
  ) {
    this.runSpeed = 6;
    this.walkSpeed = 1.5;
  }

  protected baseAccuracy(): number {
    return this.swat ? K.accuracySwat : K.accuracyBase;
  }

  get accuracy(): number {
    const ctx = this.ctx;
    return Math.min(0.95, this.baseAccuracy() + ctx.heat / K.heatDivisor + ctx.tier.accuracyBonus);
  }

  update(ped: IPed, pedCtx: PedContext, dt: number): void {
    this.arrestContact = false;
    // The ped's own knockdown timer owns this state; overriding it would stand the cop up instantly.
    if (ped.state === 'knockedDown') return;
    // `scripted` keeps the ambient spawner from recycling cops that start 150–250 m out.
    ped.state = 'scripted';
    if (ped.vehicle) {
      this.mode = 'ride';
      return;
    }
    const ctx = this.ctx;
    if (ctx.level === 0 || !ctx.playerAlive() || ctx.holdFire) {
      this.mode = 'standDown';
      ped.stop();
      ped.handsUp = false;
      return;
    }
    const player = pedCtx.playerPosition;
    const dist = pedCtx.playerDistance;
    const lethal = ctx.tier.lethal || this.provoked || this.swat || ctx.playerArmed();
    const fleeing = ctx.playerSpeed() > K.fleeSpeed || ctx.playerVehicle() !== null;

    if (!lethal && !fleeing) {
      this.arrestApproach(ped, player, dist);
      return;
    }
    if (!lethal && fleeing) {
      // Chase a running suspect; shooting starts only once provoked or the level climbs.
      this.mode = 'approach';
      ped.moveToward(player, this.runSpeed);
      ped.face(player);
      this.refreshLos(ped, dt, dist);
      if (this.hasLos) this.arrestContact = dist <= K.arrestRange;
      return;
    }
    this.combat(ped, player, dist, dt);
  }

  private arrestApproach(ped: IPed, player: V3, dist: number): void {
    ped.face(player);
    if (dist > K.arrestRange) {
      this.mode = 'approach';
      ped.moveToward(player, dist > K.approachWalkDistance ? this.runSpeed : this.walkSpeed * 1.6);
      return;
    }
    this.mode = 'arrest';
    ped.stop();
    eye.set(ped.position.x, ped.position.y + K.eyeHeight, ped.position.z);
    this.hasLos = this.ctx.sight.toPlayer(eye, K.sightRange);
    if (this.hasLos) {
      this.ctx.reportSighting('cop');
      this.arrestContact = true;
    }
  }

  private combat(ped: IPed, player: V3, dist: number, dt: number): void {
    this.mode = 'combat';
    ped.face(player);
    if (dist > K.chaseDistance || dist > K.standoffMax) {
      ped.moveToward(player, this.runSpeed);
    } else if (dist < K.standoffMin) {
      tmp.set(ped.position.x - player.x, 0, ped.position.z - player.z).normalize().multiplyScalar(4).add(ped.position);
      ped.moveToward(tmp, K.backOffSpeed);
    } else {
      this.strafeTimer -= dt;
      if (this.strafeTimer <= 0) {
        this.strafeTimer = this.ctx.rng.range(K.strafeSwitchMin, K.strafeSwitchMax);
        this.strafeDir = -this.strafeDir;
      }
      // Sidestep around the player so the cop is not a stationary target.
      tmp.set(player.z - ped.position.z, 0, -(player.x - ped.position.x)).normalize().multiplyScalar(3 * this.strafeDir).add(ped.position);
      ped.moveToward(tmp, K.strafeSpeed);
    }
    this.refreshLos(ped, dt, dist);
    if (this.hasLos && dist <= K.arrestRange) this.arrestContact = true;

    this.fireTimer -= dt;
    if (this.fireTimer > 0 || !this.hasLos || dist > K.sightRange) return;
    const cadence = COP_FIRE_INTERVAL[this.weapon];
    this.fireTimer = cadence * (1 + this.ctx.rng.range(-K.cadenceJitter, K.cadenceJitter));
    const weapons = this.ctx.game.weapons;
    if (!weapons || this.weapon === 'fists') return;
    eye.set(ped.position.x, ped.position.y + K.eyeHeight, ped.position.z);
    aimAtPlayer(this.ctx, eye, dir);
    weapons.npcFire(ped, this.weapon, eye, dir, this.accuracy);
    this.ctx.copFired(ped);
  }

  private refreshLos(ped: IPed, dt: number, dist: number): void {
    this.losTimer -= dt;
    if (this.losTimer > 0) return;
    this.losTimer = K.losInterval;
    eye.set(ped.position.x, ped.position.y + K.eyeHeight, ped.position.z);
    this.hasLos = dist <= K.sightRange && this.ctx.sight.toPlayer(eye, K.sightRange);
    if (this.hasLos) this.ctx.reportSighting('cop');
  }

  onEvent(_ped: IPed, pedEvent: PedEvent): void {
    const event = asBrainEvent(pedEvent);
    if (event.type === 'damaged' || event.type === 'hitByVehicle') {
      this.provoked = true;
      this.fireTimer = Math.min(this.fireTimer, 0.2);
    }
  }
}
