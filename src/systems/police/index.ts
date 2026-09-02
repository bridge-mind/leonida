import { Vector3 } from 'three';
import type { IPed, IVehicle, V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { IPolicePeer, WeaponId } from '../../core/peers';
import { Random } from '../../core/Random';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { DISPATCH_TABLE, dispatchTier, type DispatchTier } from '../../data/police';
import type { RoadGraph } from '../../world/RoadGraph';
import { Arrest } from './arrest';
import { PoliceConfig } from './config';
import { type PoliceContext, SightLine } from './context';
import type { CopBrain } from './CopBrain';
import { policeDebugActions } from './debug';
import { type CruiserUnit, Dispatcher } from './dispatch';
import { Helicopter } from './Helicopter';
import { planRoadblock, Roadblock, type RoadblockPlacement } from './Roadblock';
import '../wanted/events';

const S = PoliceConfig.dispatch;
const ZERO: V3 = { x: 0, y: 0, z: 0 };
const tmp = new Vector3();
const placement: RoadblockPlacement = { center: new Vector3(), roadDir: new Vector3() };

/**
 * Police response: reads the wanted level, keeps the dispatch table's unit counts alive
 * (cruisers, SWAT vans, helicopters, roadblocks), runs the arrest timer, and tears units
 * down when the level clears. Every other system is optional: without peds/vehicles the
 * system simply spawns nothing.
 */
export class PoliceSystem implements GameSystem, IPolicePeer, PoliceContext {
  readonly rng = Random.unseeded();
  readonly sight: SightLine;
  readonly cops: IPed[] = [];
  readonly brains = new Map<IPed, CopBrain>();
  readonly cruisers: CruiserUnit[] = [];
  readonly helicopters: Helicopter[] = [];
  readonly roadblocks: Roadblock[] = [];
  tier: DispatchTier = DISPATCH_TABLE[0];
  graph!: RoadGraph;
  /** Internal handles exposed for debug.ts / autotests; not part of the peer contract. */
  dispatcher!: Dispatcher;
  readonly arrest: Arrest;
  ready = false;
  private engaged = false;
  private responseTimer = 0;
  private dispatchTimer = 0;
  private roadblockCooldown = 0;
  private standDownTime = 0;
  private auditTimer = 0;
  private readonly offs: Array<() => void> = [];

  constructor(readonly game: Game) {
    this.sight = new SightLine(game);
    this.arrest = new Arrest(game);
  }

  // --- PoliceContext -------------------------------------------------------------------

  get ctx(): PoliceContext {
    return this;
  }

  get level(): number {
    return this.game.wanted?.level ?? 0;
  }

  get heat(): number {
    return this.game.wanted?.heat ?? 0;
  }

  get holdFire(): boolean {
    return this.arrest.busting || !this.playerAlive();
  }

  playerPosition(): V3 {
    return this.game.systems.has('player') ? this.game.player.position : ZERO;
  }

  playerVehicle(): IVehicle | null {
    return this.game.vehicles?.playerVehicle ?? null;
  }

  playerAlive(): boolean {
    return this.game.systems.has('player') && this.game.player.entity.isAlive;
  }

  playerSpeed(): number {
    const car = this.playerVehicle();
    if (car) return car.state.speedMs;
    if (!this.game.systems.has('player')) return 0;
    const v = this.game.player.entity.velocity;
    return Math.hypot(v.x, v.z);
  }

  playerArmed(): boolean {
    const w = this.game.weapons;
    return w !== undefined && w.current !== 'fists';
  }

  reportSighting(by: 'cop' | 'cruiser' | 'helicopter'): void {
    const p = this.playerPosition();
    this.game.events.emit('wanted:sighting', { position: { x: p.x, y: p.y, z: p.z }, by });
  }

  copFired(ped: IPed): void {
    this.game.events.emit('ped:attackedPlayer', { ped, melee: false });
  }

  registerCop(ped: IPed, brain: CopBrain): void {
    this.cops.push(ped);
    this.brains.set(ped, brain);
  }

  // --- IPolicePeer -----------------------------------------------------------------------

  get activeUnits(): number {
    return this.cruisers.length + this.helicopters.length + this.roadblocks.length;
  }

  copsNear(pos: V3, radius: number, out: IPed[]): IPed[] {
    out.length = 0;
    const r2 = radius * radius;
    for (let i = 0; i < this.cops.length; i++) {
      const c = this.cops[i];
      if (!c.isAlive) continue;
      const dx = c.position.x - pos.x;
      const dz = c.position.z - pos.z;
      if (dx * dx + dz * dz <= r2) out.push(c);
    }
    return out;
  }

  // --- Lifecycle ---------------------------------------------------------------------------

  init(): void {
    if (!this.game.systems.has('world')) {
      console.warn('[police] world system missing; police will not dispatch');
      return;
    }
    this.graph = this.game.world.roadGraph;
    this.dispatcher = new Dispatcher(this);
    this.ready = true;
    this.offs.push(
      this.game.events.on('player:respawned', () => {
        this.arrest.reset();
        this.standDownAll();
      }),
      this.game.events.on('wanted:cleared', () => this.standDownAll()),
      // PlayerControl emits this before seating the player: the crew must vacate seat 0 now, or the
      // player lands in a back seat of a cruiser nobody drives.
      this.game.events.on('vehicle:carjacked', ({ vehicle }) => this.onCarjacked(vehicle)),
    );
  }

  private onCarjacked(vehicle: IVehicle): void {
    const i = this.cruisers.findIndex((u) => u.vehicle === vehicle);
    if (i < 0) return;
    const u = this.cruisers[i];
    this.bail(u);
    this.cruisers.splice(i, 1);
  }

  fixedUpdate(dt: number): void {
    const level = this.level;
    this.tier = dispatchTier(level);
    if (!this.ready) return;

    this.auditTimer -= dt;
    if (this.auditTimer <= 0) {
      this.auditTimer = 1;
      this.audit();
    }

    if (level > 0) {
      if (!this.engaged) {
        this.engaged = true;
        this.standDownTime = 0;
        // Heat shortens the response: a known offender gets units rolling sooner.
        const k = Math.min(1, this.heat / 100);
        this.responseTimer = S.responseDelay + (S.responseDelayMinAtMaxHeat - S.responseDelay) * k;
        this.dispatchTimer = 0;
      }
      this.responseTimer -= dt;
      this.dispatchTimer -= dt;
      if (this.responseTimer <= 0 && this.dispatchTimer <= 0) {
        this.dispatchTimer = S.interval;
        this.dispatch();
      }
      this.roadblockCooldown -= dt;
      this.arrest.update(dt, level, this.cops, this.brains, this.playerSpeed(), this.playerVehicle() !== null);
    } else if (this.engaged) {
      this.engaged = false;
      this.standDownAll();
    }

    this.updateCruisers();
    const car = this.playerVehicle();
    for (let i = 0; i < this.roadblocks.length; i++) this.roadblocks[i].update(dt, car);
    for (let i = this.helicopters.length - 1; i >= 0; i--) {
      const h = this.helicopters[i];
      h.update(dt);
      if (h.finished || (h.leaving && (h.leaveTime > S.leaveHardCap || (h.leaveTime > S.leaveSeconds && !this.dispatcher.inView(h.position))))) {
        h.dispose();
        this.helicopters.splice(i, 1);
      }
    }
    if (!this.engaged) this.sweepStoodDown(dt);
    this.game.perf.set('police', this.cops.length);
  }

  dispose(): void {
    this.offs.forEach((off) => off());
    this.offs.length = 0;
    this.despawnAll();
  }

  // --- Dispatch ------------------------------------------------------------------------------

  private dispatch(): void {
    const tier = this.tier;
    const ref = this.playerPosition();
    this.dispatcher.refreshFrustum();
    this.recycleFar(ref);

    const cruisers = this.cruisers.filter((u) => u.kind === 'police').length;
    const vans = this.cruisers.filter((u) => u.kind === 'swat').length;
    if (cruisers < tier.cruisers && this.canAddUnit()) this.spawnUnit('police', tier.copWeapon);
    if (vans < tier.swatVans && this.canAddUnit()) this.spawnUnit('swat', tier.swatWeapon);
    if (this.helicopters.length < tier.helicopters && this.canAddUnit()) this.spawnHelicopter(ref);
    if (tier.roadblock && this.roadblockCooldown <= 0 && this.roadblocks.length < 2 && this.canAddUnit()) {
      if (this.spawnRoadblock(tier)) this.roadblockCooldown = PoliceConfig.roadblock.cooldown;
    }
  }

  private canAddUnit(): boolean {
    return this.activeUnits < S.maxUnits && this.cops.length < S.maxCops;
  }

  spawnUnit(kind: 'police' | 'swat', weapon: WeaponId, at?: { position: V3; heading: number }): CruiserUnit | null {
    const point = at ?? this.dispatcher.findSpawnPoint(this.playerPosition());
    if (!point) return null;
    const spawn = { position: tmp.set(point.position.x, point.position.y, point.position.z), heading: point.heading };
    const unit = this.dispatcher.spawnCruiser(kind, spawn, kind === 'swat' ? 4 : 2, weapon);
    if (unit) this.cruisers.push(unit);
    return unit;
  }

  spawnHelicopter(ref: V3): Helicopter {
    const H = PoliceConfig.helicopter;
    const a = this.rng.range(0, Math.PI * 2);
    tmp.set(ref.x + Math.cos(a) * 220, ref.y + H.altitude + 20, ref.z + Math.sin(a) * 220);
    const heli = new Helicopter(this, tmp);
    this.helicopters.push(heli);
    return heli;
  }

  spawnRoadblock(tier: DispatchTier): Roadblock | null {
    const ref = this.playerPosition();
    const car = this.playerVehicle();
    const heading = car ? car.heading : this.game.systems.has('player') ? this.game.player.heading : 0;
    const dist = this.rng.range(PoliceConfig.roadblock.aheadMin, PoliceConfig.roadblock.aheadMax);
    const plan = planRoadblock(this.graph, ref, heading, dist, placement);
    if (!plan) return null;
    const block = new Roadblock(this.dispatcher, plan, ref, tier.roadblockCops, tier.spikes, tier.copWeapon);
    this.roadblocks.push(block);
    return block;
  }

  // --- Unit upkeep -------------------------------------------------------------------------------

  private updateCruisers(): void {
    for (let i = this.cruisers.length - 1; i >= 0; i--) {
      const u = this.cruisers[i];
      const v = u.vehicle;
      if (!v.isAlive || v.state.exploded) {
        this.bail(u);
        this.cruisers.splice(i, 1);
        continue;
      }
      if (v.driverEntity?.kind === 'player') {
        // Stolen: hand the wheel to the player and let the crew out; the car is no longer ours.
        v.setDriver(null);
        this.bail(u);
        this.cruisers.splice(i, 1);
        continue;
      }
      if (v.state.onFire && !u.bailed) this.bail(u);
      if (u.driver.wantsBail) {
        u.driver.wantsBail = false;
        this.bail(u);
      }
      if (u.driver.mode === 'leave') {
        const t = u.driver.leaveSeconds;
        if (t > S.leaveHardCap || (t > S.leaveSeconds && !this.dispatcher.inView(v.position))) {
          this.removeUnit(u);
          this.cruisers.splice(i, 1);
        }
      }
    }
  }

  /** Crew leaves the car; the driver object stays so the empty cruiser holds still. */
  private bail(u: CruiserUnit): void {
    if (u.bailed) return;
    u.bailed = true;
    for (const ped of u.crew) if (ped.isAlive && ped.vehicle === u.vehicle) ped.exitVehicle();
  }

  private removeUnit(u: CruiserUnit): void {
    const peds = this.game.peds;
    for (const ped of u.crew) {
      if (ped.vehicle === u.vehicle) this.removeCop(ped, peds);
    }
    u.crew.length = 0;
    if (u.vehicle.isAlive && u.vehicle.driverEntity?.kind !== 'player') this.game.vehicles?.despawn(u.vehicle);
  }

  private removeCop(ped: IPed, peds = this.game.peds): void {
    this.brains.delete(ped);
    const idx = this.cops.indexOf(ped);
    if (idx >= 0) {
      this.cops[idx] = this.cops[this.cops.length - 1];
      this.cops.pop();
    }
    if (ped.isAlive) peds?.despawn(ped);
  }

  /** Drops cops the peds system killed or recycled, so counts and cop dots stay honest. */
  private audit(): void {
    const all = this.game.peds?.all;
    for (let i = this.cops.length - 1; i >= 0; i--) {
      const c = this.cops[i];
      if (!c.isAlive || (all && all.indexOf(c) < 0)) {
        this.brains.delete(c);
        this.cops[i] = this.cops[this.cops.length - 1];
        this.cops.pop();
      }
    }
    for (const u of this.cruisers) {
      for (let i = u.crew.length - 1; i >= 0; i--) if (!u.crew[i].isAlive) u.crew.splice(i, 1);
    }
    for (let i = this.roadblocks.length - 1; i >= 0; i--) {
      if (this.roadblocks[i].isStale(this.playerPosition())) {
        this.roadblocks[i].dispose();
        this.roadblocks.splice(i, 1);
      }
    }
  }

  /** Units far behind the chase are recycled so the dispatcher can respawn them closer. */
  private recycleFar(ref: V3): void {
    const d2 = S.recycleDistance * S.recycleDistance;
    for (let i = this.cruisers.length - 1; i >= 0; i--) {
      const u = this.cruisers[i];
      const p = u.vehicle.position;
      const dx = p.x - ref.x;
      const dz = p.z - ref.z;
      if (dx * dx + dz * dz > d2 && !this.dispatcher.inView(p)) {
        this.removeUnit(u);
        this.cruisers.splice(i, 1);
      }
    }
    for (let i = this.cops.length - 1; i >= 0; i--) {
      const c = this.cops[i];
      if (c.vehicle) continue;
      const dx = c.position.x - ref.x;
      const dz = c.position.z - ref.z;
      if (dx * dx + dz * dz > d2 && !this.dispatcher.inView(c.position)) this.removeCop(c);
    }
  }

  private standDownAll(): void {
    this.standDownTime = 0;
    for (const u of this.cruisers) {
      u.driver.standDown();
      u.vehicle.setSiren(false);
    }
    for (const h of this.helicopters) h.leaving = true;
    for (const b of this.roadblocks) b.dispose();
    this.roadblocks.length = 0;
  }

  /** After the level clears, foot cops fade once off-screen (or after the hard cap). */
  private sweepStoodDown(dt: number): void {
    if (this.cops.length === 0 && this.cruisers.length === 0 && this.helicopters.length === 0) return;
    this.standDownTime += dt;
    if (this.standDownTime < S.leaveSeconds) return;
    const force = this.standDownTime > S.leaveHardCap;
    this.dispatcher.refreshFrustum();
    for (let i = this.cops.length - 1; i >= 0; i--) {
      const c = this.cops[i];
      if (c.vehicle) continue;
      if (force || !this.dispatcher.inView(c.position)) this.removeCop(c);
    }
    if (force) {
      for (const u of this.cruisers) this.removeUnit(u);
      this.cruisers.length = 0;
    }
  }

  despawnAll(): void {
    for (const u of this.cruisers) this.removeUnit(u);
    this.cruisers.length = 0;
    for (const h of this.helicopters) h.dispose();
    this.helicopters.length = 0;
    for (const b of this.roadblocks) b.dispose();
    this.roadblocks.length = 0;
    for (let i = this.cops.length - 1; i >= 0; i--) this.removeCop(this.cops[i]);
  }

  readonly debugActions = policeDebugActions(this);
}

const module: SystemModule = {
  id: 'police',
  priority: PRIORITY.police,
  create: (game) => new PoliceSystem(game),
};

export default module;
