import { Vector3 } from 'three';
import type { IPed, PedBrain, PedContext, PedEvent, V3 } from '../../../core/entities';
import type { WeaponId } from '../../../core/peers';
import { isPedTypeId, PED_TYPES, type PedTypeDef } from '../../../data/peds';
import { PedConfig } from '../config';
import { asBrainEvent } from '../events';
import { SidewalkNav } from './SidewalkNav';

const C = PedConfig;
const tmp = new Vector3();
const tmp2 = new Vector3();
const WEAPON_IDS: readonly WeaponId[] = ['fists', 'pistol', 'smg', 'shotgun', 'rifle', 'sniper'];

/** `IPed.weaponId` is a plain string; only known ids reach `weapons.npcFire`. */
function asWeaponId(id: string | null): WeaponId | null {
  if (!id || id === 'fists') return null;
  for (let i = 0; i < WEAPON_IDS.length; i++) if (WEAPON_IDS[i] === id) return WEAPON_IDS[i];
  return null;
}

/**
 * SYSTEM_PROMPT §7 civilian state machine: WANDER along sidewalks with pauses,
 * IDLE_GROUP chatter, COWER when aimed at (robbery after 1.5 s), FLEE from threats,
 * CALL_POLICE for a tenth of witnesses, FIGHT when provoked. Gang peds reuse it with
 * `fightsBack` so any damage turns into a fight and gunfire nearby is ignored.
 */
export class CivilianBrain implements PedBrain {
  readonly nav = new SidewalkNav();
  /** The system's shared context, captured at `reset` so events can react without one. */
  private ctx: PedContext | null = null;
  private type: PedTypeDef = PED_TYPES.beachgoer;
  private pauseTimer = 0;
  private nextPauseIn = 0;
  private readonly threat = new Vector3();
  private aimAccum = 0;
  private sinceAimed = Infinity;
  private robbed = false;
  private fleeTimer = 0;
  private fleeBrief = false;
  private fleeRepick = 0;
  private callTimer = 0;
  private readonly groupCenter = new Vector3();
  private groupTimer = 0;
  private fightTimer = 0;
  private punchTimer = 0;
  private fireTimer = 0;

  constructor(readonly fightsBack = false) {}

  /** Called by the system before a pooled ped is (re)spawned with this brain. */
  reset(ped: IPed, ctx: PedContext): void {
    this.ctx = ctx;
    this.type = isPedTypeId(ped.typeId) ? PED_TYPES[ped.typeId] : PED_TYPES.beachgoer;
    this.nav.reset();
    this.pauseTimer = 0;
    this.nextPauseIn = ctx.rng.range(C.wander.pauseEveryMin, C.wander.pauseEveryMax);
    this.aimAccum = 0;
    this.sinceAimed = Infinity;
    this.robbed = false;
    this.fleeTimer = 0;
    this.fleeBrief = false;
    this.callTimer = 0;
    this.groupTimer = 0;
    this.fightTimer = 0;
    this.punchTimer = 0;
    this.fireTimer = 0;
  }

  /** Stand in a loose circle chatting for `seconds`, then wander off. */
  enterIdleGroup(ped: IPed, center: V3, seconds: number): void {
    this.groupCenter.set(center.x, center.y, center.z);
    this.groupTimer = seconds;
    ped.stop();
    ped.face(center);
    ped.state = 'idleGroup';
  }

  startFlee(ped: IPed, ctx: PedContext, threat: V3, brief = false): void {
    if (!ped.isAlive || ped.state === 'inVehicle' || ped.state === 'knockedDown') return;
    this.threat.set(threat.x, threat.y, threat.z);
    this.fleeBrief = brief;
    this.fleeTimer = brief ? C.flee.hornDuration : C.flee.duration;
    this.fleeRepick = C.flee.repick;
    ped.handsUp = false;
    this.nav.start(ctx.game.world.roadGraph, ped.position, ctx.rng, this.threat);
    if (ped.state !== 'flee') {
      ped.state = 'flee';
      ctx.game.events.emit('ped:fled', { ped });
      ctx.game.audio.play(ctx.rng.chance(0.5) ? 'ped.scream' : 'ped.scream.female', { position: ped.position });
    }
  }

  // --- Update -----------------------------------------------------------------------

  update(ped: IPed, ctx: PedContext, dt: number): void {
    switch (ped.state) {
      case 'wander':
        this.wander(ped, ctx, dt);
        break;
      case 'idleGroup':
        this.groupTimer -= dt;
        ped.stop();
        ped.face(this.groupCenter);
        if (this.groupTimer <= 0) this.toWander(ped);
        break;
      case 'cower':
        this.cower(ped, ctx, dt);
        break;
      case 'flee':
        this.flee(ped, ctx, dt);
        break;
      case 'callPolice':
        this.callTimer -= dt;
        ped.stop();
        if (this.callTimer <= 0) {
          ctx.game.events.emit('ped:reportedCrime', { ped, position: { x: ped.position.x, y: ped.position.y, z: ped.position.z } });
          this.startFlee(ped, ctx, this.threat);
        }
        break;
      case 'fight':
        this.fight(ped, ctx, dt);
        break;
      default:
        break;
    }
  }

  private toWander(ped: IPed): void {
    ped.handsUp = false;
    this.nav.reset();
    ped.state = 'wander';
  }

  private wander(ped: IPed, ctx: PedContext, dt: number): void {
    const graph = ctx.game.world.roadGraph;
    if (!this.nav.sidewalk && !this.nav.start(graph, ped.position, ctx.rng, null)) {
      ped.stop();
      return;
    }
    if (this.pauseTimer > 0) {
      this.pauseTimer -= dt;
      ped.stop();
      return;
    }
    this.nextPauseIn -= dt;
    if (this.nextPauseIn <= 0) {
      this.pauseTimer = ctx.rng.range(C.wander.pauseMin, C.wander.pauseMax);
      this.nextPauseIn = ctx.rng.range(C.wander.pauseEveryMin, C.wander.pauseEveryMax);
      return;
    }
    if (this.nav.reached(ped.position)) this.nav.advance(graph, ctx.rng, null);
    ped.moveToward(this.nav.target, this.type.walkSpeed);
  }

  private cower(ped: IPed, ctx: PedContext, dt: number): void {
    ped.stop();
    ped.handsUp = true;
    ped.face(ctx.playerPosition);
    this.sinceAimed += dt;
    if (this.sinceAimed < 0.25) this.aimAccum += dt;
    if (!this.robbed && this.aimAccum >= C.cower.robAfter) {
      this.robbed = true;
      this.rob(ped, ctx);
      this.startFlee(ped, ctx, ctx.playerPosition);
      return;
    }
    if (this.sinceAimed >= C.cower.releaseAfter) this.toWander(ped);
  }

  private rob(ped: IPed, ctx: PedContext): void {
    const amount = ctx.rng.int(C.cower.cashMin, C.cower.cashMax);
    tmp.set(Math.sin(ped.heading), 0, Math.cos(ped.heading)).multiplyScalar(C.cower.dropForward).add(ped.position);
    tmp.y += 0.3;
    ctx.game.pickups?.spawn('cash', tmp, amount);
    ctx.game.events.emit('ped:droppedCash', { ped, amount });
    ctx.game.events.emit('crime:committed', {
      type: 'rob_ped',
      position: { x: ped.position.x, y: ped.position.y, z: ped.position.z },
      actor: 'player',
      loud: false,
    });
  }

  private flee(ped: IPed, ctx: PedContext, dt: number): void {
    const graph = ctx.game.world.roadGraph;
    this.fleeTimer -= dt;
    this.fleeRepick -= dt;
    if (!this.nav.sidewalk && !this.nav.start(graph, ped.position, ctx.rng, this.threat)) {
      // No sidewalk nearby: run straight away from the threat.
      tmp.copy(ped.position).sub(this.threat).setY(0);
      if (tmp.lengthSq() < 1e-4) tmp.set(1, 0, 0);
      tmp.normalize().multiplyScalar(10).add(ped.position);
      ped.moveToward(tmp, C.flee.speed);
      return;
    }
    if (this.nav.reached(ped.position)) this.nav.advance(graph, ctx.rng, this.threat);
    if (this.fleeRepick <= 0) {
      this.fleeRepick = C.flee.repick;
      this.nav.faceAway(this.threat);
    }
    ped.moveToward(this.nav.target, Math.min(C.flee.speed, this.type.runSpeed));
    if (this.fleeTimer <= 0) {
      const dx = ped.position.x - this.threat.x;
      const dz = ped.position.z - this.threat.z;
      if (this.fleeBrief || dx * dx + dz * dz > C.flee.minDistance * C.flee.minDistance) this.toWander(ped);
    }
  }

  private enterFight(ped: IPed, ctx: PedContext): void {
    this.fightTimer = 0;
    this.punchTimer = 0.3;
    this.fireTimer = ctx.rng.range(0.2, C.fight.fireInterval);
    ped.handsUp = false;
    ped.state = 'fight';
    if (this.fightsBack) ctx.game.audio.play('gang.taunt', { position: ped.position });
  }

  private fight(ped: IPed, ctx: PedContext, dt: number): void {
    const game = ctx.game;
    this.fightTimer += dt;
    const player = game.systems.has('player') ? game.player : undefined;
    const dist = ctx.playerDistance;
    if (!player || !player.isAlive || dist > C.fight.giveUpDistance || this.fightTimer > C.fight.giveUpAfter) {
      if (this.fightsBack) this.toWander(ped);
      else this.startFlee(ped, ctx, ctx.playerPosition);
      return;
    }
    const target = ctx.playerPosition;
    const weapon = asWeaponId(ped.weaponId);
    if (weapon) {
      if (dist > C.fight.gunMaxDistance) ped.moveToward(target, this.type.runSpeed);
      else if (dist < C.fight.gunMinDistance) {
        tmp.copy(ped.position).sub(target).setY(0).normalize().multiplyScalar(5).add(ped.position);
        ped.moveToward(tmp, this.type.walkSpeed * 1.5);
      } else ped.stop();
      ped.face(target);
      this.fireTimer -= dt;
      if (this.fireTimer <= 0) {
        this.fireTimer = C.fight.fireInterval;
        tmp.set(ped.position.x, ped.position.y + C.body.eyeHeight, ped.position.z);
        tmp2.set(target.x, target.y + C.body.chestHeight, target.z).sub(tmp).normalize();
        game.weapons?.npcFire(ped, weapon, tmp, tmp2, C.fight.accuracy);
        game.events.emit('ped:attackedPlayer', { ped, melee: false });
      }
      return;
    }
    if (dist > C.fight.punchRange) {
      ped.moveToward(target, C.fight.approachSpeed);
      return;
    }
    ped.stop();
    ped.face(target);
    this.punchTimer -= dt;
    if (this.punchTimer <= 0) {
      this.punchTimer = C.fight.punchInterval;
      player.applyDamage({ amount: C.fight.punchDamage, type: 'melee', source: ped });
      game.audio.play('ped.punch', { position: ped.position });
      game.events.emit('ped:attackedPlayer', { ped, melee: true });
    }
  }

  // --- Events -----------------------------------------------------------------------

  onEvent(ped: IPed, pedEvent: PedEvent): void {
    const ctx = this.ctx;
    if (!ctx || !ped.isAlive || ped.state === 'inVehicle' || ped.state === 'dead') return;
    const event = asBrainEvent(pedEvent);
    switch (event.type) {
      case 'aimedAt':
        this.onAimedAt(ped, ctx, event.seconds);
        break;
      case 'damaged': {
        const src = event.source;
        if (src === 'player' || !src) tmp.copy(ctx.playerPosition);
        else tmp.copy(src.position);
        const fights = this.fightsBack || (event.damageType === 'melee' && ctx.rng.chance(this.type.fightChance));
        if (fights) {
          if (ped.state !== 'fight' && ped.state !== 'knockedDown') this.enterFight(ped, ctx);
        } else {
          this.startFlee(ped, ctx, tmp);
        }
        break;
      }
      case 'crimeSeen':
        if (this.fightsBack || ped.state === 'fight' || ped.state === 'flee' || ped.state === 'callPolice') break;
        this.threat.set(event.position.x, event.position.y, event.position.z);
        if (ped.state !== 'cower' && ctx.rng.chance(C.callPolice.chance)) {
          ped.stop();
          ped.handsUp = false;
          this.callTimer = C.callPolice.duration;
          ped.state = 'callPolice';
          ctx.game.audio.play('ped.phone.call', { position: ped.position });
        } else {
          this.startFlee(ped, ctx, event.position);
        }
        break;
      case 'heardGunshot':
      case 'explosion':
        if (ped.state !== 'fight') this.startFlee(ped, ctx, event.position);
        break;
      case 'horn':
        if (ped.state === 'wander' || ped.state === 'idleGroup') this.startFlee(ped, ctx, event.position, true);
        break;
      case 'hitByVehicle':
        this.startFlee(ped, ctx, ped.position);
        break;
      case 'carjacked':
        this.startFlee(ped, ctx, event.threat);
        break;
      default:
        break;
    }
  }

  private onAimedAt(ped: IPed, ctx: PedContext, seconds: number): void {
    if (ped.state === 'fight' || ped.state === 'knockedDown' || ped.state === 'callPolice') return;
    if (ped.state !== 'cower') {
      // ctx.playerDistance belongs to whichever ped updated last; measure ourselves here.
      const dx = ped.position.x - ctx.playerPosition.x;
      const dz = ped.position.z - ctx.playerPosition.z;
      if (dx * dx + dz * dz > C.cower.range * C.cower.range) return;
      this.aimAccum = 0;
      ped.stop();
      ped.handsUp = true;
      ped.state = 'cower';
      ctx.game.events.emit('ped:cowered', { ped });
      ctx.game.audio.play('ped.cower', { position: ped.position });
    }
    this.sinceAimed = 0;
    if (seconds > this.aimAccum) this.aimAccum = seconds;
  }
}
