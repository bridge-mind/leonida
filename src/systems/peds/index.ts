import type { KinematicCharacterController } from '@dimforge/rapier3d-compat';
import { Vector3 } from 'three';
import { Config } from '../../core/Config';
import type { IPed, IVehicle, PedBrain, PedContext, PedState, V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { IPedsPeer, WeaponId } from '../../core/peers';
import { Pool } from '../../core/Pool';
import { Random } from '../../core/Random';
import { SpatialHash } from '../../core/SpatialHash';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { isPedTypeId, PED_TYPE_IDS, PED_TYPES } from '../../data/peds';
import { Ped } from '../../entities/Ped';
import { CivilianBrain } from './brains/CivilianBrain';
import { GangBrain } from './brains/GangBrain';
import { PedConfig } from './config';
import { toPedEvent } from './events';
import { PedRenderer } from './PedRenderer';
import { PedSpawner } from './PedSpawner';
import './events';

const SIM = PedConfig.sim;
const RD = PedConfig.ragdoll;
const ZERO = new Vector3();
const tmpV = new Vector3();

interface BrainSet {
  civilian: CivilianBrain;
  gang: GangBrain;
}

/**
 * Pedestrians: pooled `Ped` entities, one shared character controller, instanced
 * rendering, ambient spawning, spatial hash queries and the cross-system event hooks
 * (aimed at, crimes, explosions, horns, carjacks, vehicle contacts).
 */
export class PedSystem implements GameSystem, IPedsPeer {
  readonly typeIds: readonly string[] = PED_TYPE_IDS;
  /** Live peds (spawned, not yet despawned). */
  readonly live: Ped[] = [];
  readonly rng = Random.unseeded();
  readonly hash = new SpatialHash<Ped>(SIM.spatialCell);
  renderer!: PedRenderer;
  spawner!: PedSpawner;
  ctx!: PedContext;
  private controller!: KinematicCharacterController;
  private readonly pool: Pool<Ped>;
  private readonly brainSets = new Map<Ped, BrainSet>();
  private readonly queryBuf: Ped[] = [];
  private readonly offs: Array<() => void> = [];
  private tick = 0;
  private debugAimPed: Ped | null = null;
  private debugAimTimer = 0;
  private debugAimSeconds = 0;

  constructor(private readonly game: Game) {
    this.pool = new Pool<Ped>(() => {
      const ped = new Ped(game, this.rng);
      this.brainSets.set(ped, { civilian: new CivilianBrain(), gang: new GangBrain() });
      return ped;
    });
  }

  init(): void {
    const game = this.game;
    const M = PedConfig.motion;
    this.controller = game.physics.world.createCharacterController(Config.physics.characterOffset);
    this.controller.setUp({ x: 0, y: 1, z: 0 });
    this.controller.enableAutostep(M.autostepHeight, M.autostepMinWidth, true);
    this.controller.enableSnapToGround(M.snapToGround);
    this.controller.setApplyImpulsesToDynamicBodies(false);
    this.controller.setMaxSlopeClimbAngle((50 * Math.PI) / 180);
    this.controller.setMinSlopeSlideAngle((55 * Math.PI) / 180);
    this.controller.setSlideEnabled(true);
    this.renderer = new PedRenderer(game.actorsRoot);
    this.spawner = new PedSpawner(game, this, this.rng);
    const playerPos = game.systems.has('player') ? game.player.position : ZERO;
    this.ctx = { game, playerPosition: playerPos, playerDistance: 0, peds: this, rng: this.rng, isNight: game.time.isNight };
    this.subscribe();
  }

  get all(): readonly IPed[] {
    return this.live;
  }

  // --- IPedsPeer --------------------------------------------------------------------

  spawn(typeId: string, position: V3, opts?: { brain?: PedBrain; heading?: number; weaponId?: WeaponId }): IPed {
    const type = isPedTypeId(typeId) ? PED_TYPES[typeId] : PED_TYPES.tourist;
    if (this.live.length >= this.renderer.capacity) this.despawn(this.farthest());
    const ped = this.pool.acquire();
    const brains = this.brainSets.get(ped) as BrainSet;
    const palette = this.rng.pick(type.palettes);
    const brain = opts?.brain ?? (type.id === 'gang' ? brains.gang : brains.civilian);
    const ground = this.game.physics.groundHeightAt(position.x, position.z, position.y + 3, 12);
    tmpV.set(position.x, (ground ?? position.y) + 0.05, position.z);
    ped.spawn(type, palette, tmpV, opts?.heading ?? this.rng.range(0, Math.PI * 2), brain);
    ped.instanceIndex = this.renderer.acquire(palette);
    if (opts?.weaponId !== undefined) ped.weaponId = opts.weaponId;
    else if (type.id === 'gang') ped.weaponId = this.rng.chance(PedConfig.gang.pistolChance) ? 'pistol' : null;
    if (brain instanceof CivilianBrain) brain.reset(ped, this.ctx);
    this.live.push(ped);
    this.hash.insert(ped, ped.position.x, ped.position.z);
    this.game.events.emit('ped:spawned', { ped });
    return ped;
  }

  despawn(ped: IPed): void {
    if (!(ped instanceof Ped) || !ped.active) return;
    this.renderer.release(ped.instanceIndex);
    ped.instanceIndex = -1;
    this.hash.remove(ped);
    const idx = this.live.indexOf(ped);
    if (idx >= 0) {
      this.live[idx] = this.live[this.live.length - 1];
      this.live.pop();
    }
    if (this.debugAimPed === ped) this.debugAimPed = null;
    ped.despawn();
    this.pool.release(ped);
  }

  query(pos: V3, radius: number, out: IPed[], filter?: (p: IPed) => boolean): IPed[] {
    const buf = this.hash.query(pos.x, pos.z, radius, this.queryBuf);
    out.length = 0;
    for (let i = 0; i < buf.length; i++) {
      const p = buf[i];
      if (!p.active) continue;
      if (filter && !filter(p)) continue;
      out.push(p);
    }
    return out;
  }

  witnessesNear(pos: V3, radius: number, out: IPed[]): IPed[] {
    const buf = this.hash.query(pos.x, pos.z, radius, this.queryBuf);
    out.length = 0;
    for (let i = 0; i < buf.length; i++) {
      const p = buf[i];
      if (p.active && p.isAlive && !p.vehicle && p.state !== 'knockedDown') out.push(p);
    }
    return out;
  }

  /** Live ped by entity id (autotests, debug). */
  byId(id: number): Ped | undefined {
    for (let i = 0; i < this.live.length; i++) if (this.live[i].id === id) return this.live[i];
    return undefined;
  }

  nearest(pos: V3, filter?: (p: Ped) => boolean): Ped | null {
    let best: Ped | null = null;
    let bestD = Infinity;
    for (let i = 0; i < this.live.length; i++) {
      const p = this.live[i];
      if (filter && !filter(p)) continue;
      const dx = p.position.x - pos.x;
      const dz = p.position.z - pos.z;
      const d = dx * dx + dz * dz;
      if (d < bestD) {
        bestD = d;
        best = p;
      }
    }
    return best;
  }

  private farthest(): Ped {
    const pos = this.ctx.playerPosition;
    let best = this.live[0];
    let bestD = -1;
    for (let i = 0; i < this.live.length; i++) {
      const p = this.live[i];
      const dx = p.position.x - pos.x;
      const dz = p.position.z - pos.z;
      const d = dx * dx + dz * dz + (p.isAlive ? 0 : 1e6);
      if (d > bestD) {
        bestD = d;
        best = p;
      }
    }
    return best;
  }

  // --- Loop -------------------------------------------------------------------------

  fixedUpdate(dt: number): void {
    this.tick++;
    const ctx = this.ctx;
    ctx.isNight = this.game.time.isNight;
    const pp = ctx.playerPosition;
    for (let i = this.live.length - 1; i >= 0; i--) {
      const ped = this.live[i];
      if (ped.wantsDespawn) {
        this.despawn(ped);
        continue;
      }
      const dx = ped.position.x - pp.x;
      const dz = ped.position.z - pp.z;
      const dist = Math.sqrt(dx * dx + dz * dz);
      let stepDt = dt;
      if (dist > SIM.nearRadius) {
        if (dist > SIM.farRadius && !ped.isAlive) {
          this.despawn(ped);
          continue;
        }
        if ((this.tick + ped.id) % SIM.farStepEvery !== 0) continue;
        stepDt = dt * SIM.farStepEvery;
      }
      ctx.playerDistance = dist;
      if (ped.isAlive && !ped.ragdolled) ped.brain.update(ped, ctx, stepDt);
      ped.fixedUpdate(stepDt, this.controller);
      this.hash.update(ped, ped.position.x, ped.position.z);
    }
  }

  update(dt: number, alpha: number): void {
    const pp = this.ctx.playerPosition;
    if (this.game.systems.has('player')) this.spawner.update(dt, this.game.player.position);
    const animR2 = SIM.animRadius * SIM.animRadius;
    for (let i = 0; i < this.live.length; i++) {
      const ped = this.live[i];
      const dx = ped.position.x - pp.x;
      const dz = ped.position.z - pp.z;
      ped.updateVisual(dt, alpha, dx * dx + dz * dz <= animR2);
      this.renderer.write(ped);
    }
    this.renderer.commit();
    this.game.perf.set('peds', this.live.length);
    this.updateDebugAim(dt);
  }

  private updateDebugAim(dt: number): void {
    const ped = this.debugAimPed;
    if (!ped) return;
    if (this.debugAimTimer <= 0 || !ped.active) {
      this.debugAimPed = null;
      return;
    }
    this.debugAimTimer -= dt;
    this.debugAimSeconds += dt;
    this.game.events.emit('weapon:aimedAt', { target: ped, seconds: this.debugAimSeconds });
  }

  // --- Cross-system hooks -----------------------------------------------------------

  private subscribe(): void {
    const ev = this.game.events;
    this.offs.push(
      ev.on('weapon:aimedAt', ({ target, seconds }) => {
        if (target instanceof Ped && target.active) target.notify(toPedEvent({ type: 'aimedAt', seconds }));
      }),
      ev.on('crime:committed', ({ position, loud }) => {
        const radius = loud ? PedConfig.callPolice.loudRange : PedConfig.callPolice.witnessRange;
        this.broadcast(position, radius, (distance) => toPedEvent({ type: 'crimeSeen', position, distance, loud }));
      }),
      ev.on('vehicle:exploded', ({ position }) => {
        this.broadcast(position, PedConfig.flee.explosionRange, (distance) => toPedEvent({ type: 'explosion', position, distance }));
      }),
      ev.on('vehicle:horn', ({ vehicle }) => this.onHorn(vehicle)),
      ev.on('vehicle:carjacked', ({ vehicle, position, heading, driverTypeId }) => {
        // A real crew (police cruisers) bails on this same event; only virtual traffic drivers need a ped conjured.
        if (vehicle.state.occupants.some((o) => o.kind === 'ped')) return;
        const ped = this.spawn(driverTypeId, position, { heading }) as Ped;
        ped.notify(toPedEvent({ type: 'carjacked', threat: this.ctx.playerPosition }));
      }),
      ev.on('physics:contactStarted', ({ a, b }) => {
        if (a instanceof Ped && b && b.kind === 'vehicle') this.onVehicleContact(a, b as IVehicle);
        else if (b instanceof Ped && a && a.kind === 'vehicle') this.onVehicleContact(b, a as IVehicle);
      }),
    );
  }

  private broadcast(position: V3, radius: number, make: (distance: number) => ReturnType<typeof toPedEvent>, maxDy = 12): void {
    const buf = this.hash.query(position.x, position.z, radius, this.queryBuf);
    for (let i = 0; i < buf.length; i++) {
      const ped = buf[i];
      if (!ped.active || !ped.isAlive || ped.vehicle) continue;
      // Interiors sit far above the city on the same x/z; a shot indoors must not alarm the street.
      if (Math.abs(ped.position.y - position.y) > maxDy) continue;
      const dx = ped.position.x - position.x;
      const dz = ped.position.z - position.z;
      ped.notify(make(Math.sqrt(dx * dx + dz * dz)));
    }
  }

  private onHorn(vehicle: IVehicle): void {
    const vp = vehicle.position;
    const fx = Math.sin(vehicle.heading);
    const fz = Math.cos(vehicle.heading);
    const buf = this.hash.query(vp.x, vp.z, PedConfig.flee.hornRange, this.queryBuf);
    for (let i = 0; i < buf.length; i++) {
      const ped = buf[i];
      if (!ped.active || !ped.isAlive || ped.vehicle) continue;
      // Only peds in front of the car react; the horn is aimed at them.
      if ((ped.position.x - vp.x) * fx + (ped.position.z - vp.z) * fz <= 0) continue;
      ped.notify(toPedEvent({ type: 'horn', position: vp }));
    }
  }

  private onVehicleContact(ped: Ped, vehicle: IVehicle): void {
    if (!ped.isAlive || ped.ragdolled || ped.vehicle === vehicle || ped.state === 'knockedDown') return;
    const speed = vehicle.state.speedMs;
    if (speed >= RD.vehicleSpeedThreshold) {
      const v = vehicle.velocity;
      tmpV.set(v.x * RD.impulsePerMs, v.y * RD.impulsePerMs + RD.upImpulse, v.z * RD.impulsePerMs);
      ped.ragdoll(tmpV);
      this.game.events.emit('ped:knockedDown', { ped, source: vehicle });
      const len = Math.hypot(v.x, v.z);
      // Through the damage system so a fatal run-over emits `entity:killed` / `kill_ped`
      // (attributed to the player when they are driving), like every other kill.
      const info = {
        type: 'vehicle' as const,
        amount: speed * RD.damagePerMs,
        source: vehicle.driverEntity?.kind === 'player' ? ('player' as const) : vehicle,
        direction: len > 1e-3 ? { x: v.x / len, y: 0, z: v.z / len } : undefined,
      };
      if (!this.game.damage?.apply(ped, info)) ped.applyDamage(info);
    } else {
      ped.knockDown(vehicle, speed);
    }
  }

  dispose(): void {
    for (const off of this.offs) off();
    this.offs.length = 0;
    for (let i = this.live.length - 1; i >= 0; i--) this.despawn(this.live[i]);
    for (const ped of this.brainSets.keys()) ped.dispose();
    this.brainSets.clear();
    this.renderer.dispose();
    this.game.physics.world.removeCharacterController(this.controller);
  }

  // --- Debug ------------------------------------------------------------------------

  private playerAhead(meters: number): V3 {
    const player = this.game.player;
    const h = player.heading;
    return { x: player.position.x + Math.sin(h) * meters, y: player.position.y, z: player.position.z + Math.cos(h) * meters };
  }

  private nearestToPlayer(): Ped | null {
    return this.nearest(this.ctx.playerPosition, (p) => p.isAlive);
  }

  debugActions = {
    spawn: (type: unknown, x: unknown, z: unknown): number => this.spawn(String(type), { x: Number(x), y: 0, z: Number(z) }).id,
    spawnHere: (type: unknown): number => {
      const pos = this.playerAhead(3);
      const ped = this.spawn(String(type), pos, { heading: this.game.player.heading + Math.PI });
      return ped.id;
    },
    count: (): number => this.live.length,
    /** Simulates `weapon:aimedAt` on a ped (nearest alive, or the given id) for 2 s. */
    aimAt: (id?: unknown): number | null => {
      const ped = id === undefined ? this.nearestToPlayer() : (this.byId(Number(id)) ?? null);
      if (!ped) return null;
      this.debugAimPed = ped;
      this.debugAimTimer = PedConfig.debug.aimSeconds;
      this.debugAimSeconds = 0;
      return ped.id;
    },
    gunshot: (): void => {
      const p = this.ctx.playerPosition;
      this.game.events.emit('crime:committed', { type: 'shot_fired', position: { x: p.x, y: p.y, z: p.z }, actor: 'player', loud: true });
    },
    ragdollNearest: (): number | null => {
      const ped = this.nearestToPlayer();
      if (!ped) return null;
      ped.forward(tmpV).multiplyScalar(RD.deathImpulse);
      tmpV.y = RD.upImpulse;
      ped.ragdoll(tmpV);
      return ped.id;
    },
    killNearest: (): number | null => {
      const ped = this.nearestToPlayer();
      if (!ped) return null;
      ped.kill('player');
      return ped.id;
    },
    setDensity: (n: unknown): number | null => {
      const v = Number(n);
      this.spawner.densityOverride = Number.isFinite(v) && v >= 0 ? Math.round(v) : null;
      return this.spawner.densityOverride;
    },
    stats: (): Record<string, number> => {
      const out: Record<string, number> = { total: this.live.length, target: this.spawner.targetCount(this.game.player.position) };
      for (const ped of this.live) {
        const key: PedState = ped.state;
        out[key] = (out[key] ?? 0) + 1;
      }
      return out;
    },
    state: (id: unknown): { id: number; state: PedState; typeId: string; health: number; alive: boolean } | null => {
      const ped = this.byId(Number(id));
      return ped ? { id: ped.id, state: ped.state, typeId: ped.typeId, health: ped.health, alive: ped.isAlive } : null;
    },
    damage: (id: unknown, amount: unknown, type: unknown): number | null => {
      const ped = this.byId(Number(id));
      if (!ped) return null;
      const kind = String(type ?? 'bullet') as 'bullet' | 'melee' | 'explosion' | 'vehicle' | 'fall' | 'fire';
      ped.applyDamage({ amount: Number(amount), type: kind, source: 'player' });
      return ped.health;
    },
    ragdoll: (id: unknown): boolean => {
      const ped = this.byId(Number(id));
      if (!ped) return false;
      ped.forward(tmpV).multiplyScalar(RD.deathImpulse);
      tmpV.y = RD.upImpulse;
      ped.ragdoll(tmpV);
      return true;
    },
    /** Advances every ragdoll's freeze/despawn timer (autotests avoid waiting 23 s). */
    fastForward: (seconds: unknown): number => {
      const s = Number(seconds);
      let n = 0;
      for (const ped of this.live) {
        if (ped.ragdolled) {
          ped.ragdollTime += s;
          n++;
        }
      }
      return n;
    },
  };
}

const module: SystemModule = {
  id: 'peds',
  priority: PRIORITY.peds,
  create: (game) => new PedSystem(game),
};

export default module;
