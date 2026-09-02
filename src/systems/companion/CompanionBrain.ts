import { Vector3 } from 'three';
import type { IPed, IVehicle, PedBrain, PedContext, PedEvent, V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { RaycastHit } from '../../physics/Physics';
import { asBrainEvent } from '../peds/events';
import { CompanionConfig, teleportPed } from './config';

const C = CompanionConfig;
const tmp = new Vector3();
const tmp2 = new Vector3();
const door = new Vector3();

/**
 * Jason/Lucia as an NPC: shadows the player, rides shotgun, shoots whoever is shooting
 * at the player and cannot die (a depleted pool knocks them down for a while instead).
 */
export class CompanionBrain implements PedBrain {
  hp: number = C.hitPoints;
  downed = 0;
  private fireTimer = 0;
  private readonly buf: IPed[] = [];
  private readonly hit: RaycastHit = { entity: null, point: new Vector3(), normal: new Vector3(), toi: 0, collider: null as unknown as RaycastHit['collider'] };

  constructor(
    private readonly game: Game,
    /** Peds that recently attacked the player (timestamps in real seconds). */
    private readonly hostiles: Map<IPed, number>,
  ) {}

  reset(): void {
    this.hp = C.hitPoints;
    this.downed = 0;
    this.fireTimer = 0;
  }

  get isDowned(): boolean {
    return this.downed > 0;
  }

  update(ped: IPed, ctx: PedContext, dt: number): void {
    // Pinned health: damage is tracked in `hp` (see onEvent) so a single hit can never kill.
    ped.health = C.pinnedHealth;
    if (!ped.vehicle) ped.state = 'scripted';
    const game = this.game;
    if (this.downed > 0) {
      this.downed -= dt;
      ped.stop();
      ped.handsUp = true;
      if (this.downed <= 0) {
        ped.handsUp = false;
        this.hp = C.hitPoints;
      }
      return;
    }
    if (!game.systems.has('player')) return;
    if (game.state.interior) {
      ped.stop();
      return;
    }
    const player = game.player;
    const pv = player.mountedVehicle as IVehicle | null;
    if (pv) {
      this.rideAlong(ped, pv, dt);
      return;
    }
    if (ped.vehicle) {
      ped.exitVehicle();
      return;
    }
    const target = this.pickHostile(ped);
    this.follow(ped, ctx.playerPosition, ctx.playerDistance, target !== null);
    if (target) this.engage(ped, target, dt);
  }

  private follow(ped: IPed, playerPos: V3, dist: number, holding: boolean): void {
    const F = C.follow;
    if (dist > F.teleportDistance) {
      tmp.set(playerPos.x - ped.position.x, 0, playerPos.z - ped.position.z).normalize().multiplyScalar(-F.distance).add(playerPos);
      teleportPed(ped, tmp);
      return;
    }
    if (dist <= F.stopDistance || (holding && dist < C.combat.range * 0.6)) {
      ped.stop();
      if (!holding) ped.face(playerPos);
      return;
    }
    const speed = dist > F.runDistance ? this.runSpeed(ped) : this.walkSpeed(ped);
    tmp.set(playerPos.x - ped.position.x, 0, playerPos.z - ped.position.z);
    const len = tmp.length();
    if (len > 1e-3) tmp.multiplyScalar((len - F.distance) / len);
    tmp.add(ped.position);
    ped.moveToward(tmp, speed);
  }

  private rideAlong(ped: IPed, v: IVehicle, dt: number): void {
    if (ped.vehicle === v) {
      this.shootFromCar(ped, v, dt);
      return;
    }
    if (ped.vehicle) ped.exitVehicle();
    v.seatDoorPosition(C.vehicle.preferredSeat, door);
    const d = Math.hypot(door.x - ped.position.x, door.z - ped.position.z);
    if (d > C.vehicle.catchUpDistance) teleportPed(ped, door);
    if (d <= C.vehicle.enterDistance || d > C.vehicle.catchUpDistance) {
      if (!ped.enterVehicle(v, C.vehicle.preferredSeat)) ped.enterVehicle(v);
      return;
    }
    ped.moveToward(door, this.runSpeed(ped));
  }

  private shootFromCar(ped: IPed, v: IVehicle, dt: number): void {
    const wanted = this.game.wanted;
    const police = this.game.police;
    if (!wanted || wanted.level <= 0 || !police) return;
    police.copsNear(v.position, C.combat.carRange, this.buf);
    let best: IPed | null = null;
    let bestD = Infinity;
    for (let i = 0; i < this.buf.length; i++) {
      const cop = this.buf[i];
      if (!cop.isAlive) continue;
      const d = cop.position.distanceToSquared(v.position);
      if (d < bestD) {
        bestD = d;
        best = cop;
      }
    }
    if (!best) return;
    this.fireTimer -= dt;
    if (this.fireTimer > 0) return;
    this.fireTimer = 1 / C.combat.shotsPerSecond;
    tmp.set(v.position.x, v.position.y + 1.1, v.position.z);
    tmp2.set(best.position.x, best.position.y + C.combat.chestHeight, best.position.z).sub(tmp).normalize();
    this.game.weapons?.npcFire(ped, C.combat.weapon, tmp, tmp2, C.combat.accuracy);
  }

  private pickHostile(ped: IPed): IPed | null {
    const peds = this.game.peds;
    if (!peds) return null;
    const now = this.game.time.realElapsed;
    const copsHostile = (this.game.wanted?.level ?? 0) > 0;
    peds.query(ped.position, C.combat.range, this.buf);
    let best: IPed | null = null;
    let bestD = Infinity;
    for (let i = 0; i < this.buf.length; i++) {
      const p = this.buf[i];
      if (p === ped || !p.isAlive || p.vehicle) continue;
      if (!this.isHostile(p, now, copsHostile)) continue;
      const d = p.position.distanceToSquared(ped.position);
      if (d < bestD) {
        bestD = d;
        best = p;
      }
    }
    return best;
  }

  private isHostile(p: IPed, now: number, copsHostile: boolean): boolean {
    if (p.isCop) return copsHostile;
    const t = this.hostiles.get(p);
    if (t !== undefined && now - t < C.combat.hostileMemory) return true;
    return p.state === 'fight' && C.combat.gangTypes.includes(p.typeId);
  }

  private engage(ped: IPed, target: IPed, dt: number): void {
    ped.face(target.position);
    this.fireTimer -= dt;
    if (this.fireTimer > 0) return;
    this.fireTimer = 1 / C.combat.shotsPerSecond;
    tmp.set(ped.position.x, ped.position.y + C.combat.eyeHeight, ped.position.z);
    tmp2.set(target.position.x, target.position.y + C.combat.chestHeight, target.position.z).sub(tmp);
    const dist = tmp2.length();
    tmp2.multiplyScalar(1 / Math.max(1e-6, dist));
    const hit = this.game.physics.raycastInto(this.hit, tmp, tmp2, dist, { exclude: ped });
    if (hit && hit.entity !== target && hit.toi < dist - 0.6) return;
    this.game.weapons?.npcFire(ped, C.combat.weapon, tmp, tmp2, C.combat.accuracy);
  }

  onEvent(ped: IPed, pedEvent: PedEvent): void {
    const event = asBrainEvent(pedEvent);
    if (event.type !== 'damaged' || this.downed > 0) return;
    this.hp -= event.amount;
    ped.health = C.pinnedHealth;
    if (this.hp <= 0) {
      this.downed = C.downedSeconds;
      ped.stop();
      const dir = event.direction;
      tmp.set(dir?.x ?? 0, 0, dir?.z ?? 1).normalize().multiplyScalar(60);
      tmp.y = 40;
      ped.ragdoll(tmp);
    }
  }

  private runSpeed(ped: IPed): number {
    return ped.typeId === 'lucia' || ped.typeId === 'jason' ? 7 : 5.5;
  }

  private walkSpeed(ped: IPed): number {
    return ped.typeId === 'lucia' || ped.typeId === 'jason' ? 2.4 : 1.6;
  }
}
