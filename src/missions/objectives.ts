import { Vector3 } from 'three';
import type { Entity, IVehicle, V3 } from '../core/entities';
import type { EventName, GameEvents } from '../core/Events';
import type { BlipKind } from '../core/GameState';
import type { MissionContext, Objective, ObjectiveStatus } from './types';

/**
 * Objective primitives (SYSTEM_PROMPT §14). Each factory returns a fresh object whose
 * state is private to that mission attempt. `skip` is the debug fast-forward: it forces
 * the world into the done state (teleport, mount, clear) so `update` reports 'done'.
 */

export function horizontal(a: V3, b: V3): number {
  const dx = a.x - b.x;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dz * dz);
}

const NO_TARGET = (): V3 | null => null;
const tmpDoor = new Vector3();

export function playerPos(ctx: MissionContext): V3 | null {
  return ctx.game.systems.has('player') ? ctx.game.player.position : null;
}

/** Vehicle the player is riding in, from the player entity (works for scripted mounts too). */
export function playerVehicle(ctx: MissionContext): IVehicle | null {
  if (!ctx.game.systems.has('player')) return null;
  return (ctx.game.player.mountedVehicle as IVehicle | null) ?? null;
}

export function vehicleGone(v: IVehicle | null): boolean {
  return !v || !v.isAlive || v.state.wrecked || v.state.exploded;
}

function headingTo(from: V3, to: V3): number {
  return Math.atan2(to.x - from.x, to.z - from.z);
}

/** Teleports the player, or the car they sit in so the camera and seat stay consistent. */
export function teleportPlayer(ctx: MissionContext, pos: V3, heading?: number): void {
  const game = ctx.game;
  if (!game.systems.has('player')) return;
  const v = playerVehicle(ctx);
  if (v) {
    v.teleport({ x: pos.x, y: pos.y, z: pos.z }, heading ?? v.heading);
    if (game.systems.has('camera')) game.camera.snap();
  } else {
    game.player.teleport(pos, heading);
  }
}

/**
 * Puts the player behind the wheel of `v` at once (debug fast-forward). Any scripted
 * driver or occupant is evicted first; PlayerControl adopts a player-driven car on its next tick.
 */
export function forceMount(ctx: MissionContext, v: IVehicle): void {
  const game = ctx.game;
  if (!game.systems.has('player')) return;
  const player = game.player;
  if (player.mountedVehicle === v) return;
  if (player.isMounted) game.vehicles?.exitCurrent();
  const door = v.seatDoorPosition(0, tmpDoor);
  player.teleport({ x: door.x, y: door.y, z: door.z }, v.heading, false);
  v.setDriver(null);
  const occupant = v.driverEntity;
  if (occupant && occupant.kind === 'ped') (occupant as { exitVehicle?: () => void }).exitVehicle?.();
  const seat = v.enter(player.entity, 0);
  if (seat >= 0) player.mount(v, seat);
}

// --- Primitives -----------------------------------------------------------------------

export interface GotoOptions {
  /** Player must arrive in this vehicle; losing it fails the objective. */
  vehicle?: () => IVehicle | null;
  blip?: BlipKind;
}

export function goto(pos: V3, radius: number, text: string, opts: GotoOptions = {}): Objective {
  const target = { x: pos.x, y: pos.y, z: pos.z };
  return {
    text,
    blipKind: opts.blip ?? 'objective',
    target: () => target,
    start() {},
    update(ctx) {
      const p = playerPos(ctx);
      if (!p) return 'running';
      if (opts.vehicle) {
        const v = opts.vehicle();
        if (vehicleGone(v)) {
          ctx.fail('vehicle destroyed');
          return 'failed';
        }
        if (playerVehicle(ctx) !== v) return 'running';
      }
      return horizontal(p, target) <= radius ? 'done' : 'running';
    },
    skip(ctx) {
      if (opts.vehicle) {
        const v = opts.vehicle();
        if (v) forceMount(ctx, v);
      }
      teleportPlayer(ctx, target);
    },
    stop() {},
  };
}

export function enterVehicle(ref: () => IVehicle | null, text: string): Objective {
  return {
    text,
    blipKind: 'vehicle',
    target: () => ref()?.position ?? null,
    start() {},
    update(ctx) {
      const v = ref();
      if (vehicleGone(v)) {
        ctx.fail('vehicle destroyed');
        return 'failed';
      }
      return playerVehicle(ctx) === v ? 'done' : 'running';
    },
    skip(ctx) {
      const v = ref();
      if (v) forceMount(ctx, v);
    },
    stop() {},
  };
}

/** Drive `ref` into a radius around `pos`; fails when the vehicle is lost. */
export function deliver(ref: () => IVehicle | null, pos: V3, radius: number, text: string): Objective {
  return goto(pos, radius, text, { vehicle: ref });
}

/**
 * Peds and vehicles are pooled: a target that died (or was culled) can come back as a
 * fresh ambient entity with `isAlive === true` behind the mission's reference. The tracker
 * remembers which targets are gone for good — killed (`ped:died` / `entity:killed`) or
 * recycled by a later `ped:spawned` — so objectives never wait on a stranger.
 */
export class TargetTracker {
  private readonly gone = new Set<Entity>();
  private readonly died = new Set<Entity>();
  private offs: (() => void)[] = [];

  constructor(private readonly targets: () => readonly Entity[]) {}

  start(ctx: MissionContext): void {
    this.gone.clear();
    this.died.clear();
    const ev = ctx.game.events;
    const has = (e: Entity): boolean => this.targets().includes(e);
    this.offs = [
      ev.on('ped:died', ({ ped }) => {
        if (has(ped)) this.markDead(ped);
      }),
      ev.on('entity:killed', ({ target }) => {
        if (has(target)) this.markDead(target);
      }),
      ev.on('ped:spawned', ({ ped }) => {
        if (has(ped)) this.gone.add(ped);
      }),
    ];
  }

  stop(): void {
    for (const off of this.offs) off();
    this.offs = [];
  }

  private markDead(e: Entity): void {
    this.died.add(e);
    this.gone.add(e);
  }

  /** Target is alive and still the entity the mission spawned. */
  isLive(e: Entity): boolean {
    return e.isAlive && !this.gone.has(e);
  }

  /** Target was killed (not merely despawned or recycled). */
  wasKilled(e: Entity): boolean {
    return this.died.has(e) || (e as Entity & { state?: string }).state === 'dead';
  }

  firstLive(): Entity | null {
    const list = this.targets();
    for (let i = 0; i < list.length; i++) if (this.isLive(list[i])) return list[i];
    return null;
  }

  allGone(): boolean {
    return this.firstLive() === null;
  }
}

export function kill(targets: () => readonly Entity[], text: string): Objective {
  const tracker = new TargetTracker(targets);
  return {
    text,
    blipKind: 'objective',
    target: () => tracker.firstLive()?.position ?? null,
    start(ctx) {
      tracker.start(ctx);
    },
    update: () => (tracker.allGone() ? 'done' : 'running'),
    skip() {
      const list = targets();
      for (const e of list) {
        const k = e as Entity & { kill?: () => void };
        if (tracker.isLive(e) && typeof k.kill === 'function') k.kill();
      }
    },
    stop() {
      tracker.stop();
    },
  };
}

export interface SurviveOptions {
  /** Ends early once every one of these is dead. */
  orKill?: () => readonly Entity[];
  /** Live countdown appended to the HUD text. */
  showTimer?: boolean;
}

export function survive(seconds: number, text: string, opts: SurviveOptions = {}): Objective {
  let elapsed = 0;
  const tracker = opts.orKill ? new TargetTracker(opts.orKill) : null;
  const obj: Objective = {
    text,
    blipKind: 'objective',
    target: NO_TARGET,
    start(ctx) {
      elapsed = 0;
      tracker?.start(ctx);
    },
    update(_ctx, dt) {
      elapsed += dt;
      if (opts.showTimer) obj.text = `${text} ~y~${Math.max(0, Math.ceil(seconds - elapsed))}s~y~`;
      if (tracker && tracker.allGone()) return 'done';
      return elapsed >= seconds ? 'done' : 'running';
    },
    skip() {
      elapsed = seconds;
    },
    stop() {
      tracker?.stop();
    },
  };
  return obj;
}

export function waitFor<K extends EventName>(event: K, predicate: (payload: GameEvents[K], ctx: MissionContext) => boolean, text: string): Objective {
  let hit = false;
  let off: (() => void) | null = null;
  return {
    text,
    blipKind: 'objective',
    target: NO_TARGET,
    start(ctx) {
      hit = false;
      off = ctx.game.events.on(event, (payload) => {
        if (!hit && predicate(payload, ctx)) hit = true;
      });
    },
    update: () => (hit ? 'done' : 'running'),
    skip() {
      hit = true;
    },
    stop() {
      off?.();
      off = null;
    },
  };
}

export interface CustomSpec {
  text: string;
  blipKind?: BlipKind;
  target?: () => V3 | null;
  start?(ctx: MissionContext): void;
  update(ctx: MissionContext, dt: number): ObjectiveStatus;
  skip(ctx: MissionContext): void;
  stop?(ctx: MissionContext): void;
}

export function custom(spec: CustomSpec): Objective {
  return {
    text: spec.text,
    blipKind: spec.blipKind ?? 'objective',
    target: spec.target ?? NO_TARGET,
    start: spec.start ?? (() => {}),
    update: spec.update,
    skip: spec.skip,
    stop: spec.stop ?? (() => {}),
  };
}

/**
 * Runs `fn` once; done when its promise settles. Used for spawns, cutscenes and
 * scripted beats. `skip` resolves at once (the promise is left to settle on its own).
 */
export function action(fn: (ctx: MissionContext) => void | Promise<void>, text = '', onSkip?: (ctx: MissionContext) => void): Objective {
  let finished = false;
  return {
    text,
    blipKind: 'objective',
    target: NO_TARGET,
    start(ctx) {
      finished = false;
      const result = fn(ctx);
      if (result && typeof result.then === 'function') {
        result.then(
          () => {
            finished = true;
          },
          (err: unknown) => {
            console.error(`[missions:${ctx.missionId}] action failed`, err);
            finished = true;
          },
        );
      } else finished = true;
    },
    update: () => (finished ? 'done' : 'running'),
    skip(ctx) {
      onSkip?.(ctx);
      finished = true;
    },
    stop() {},
  };
}

/** Fails the wrapped objective when it is not done within `seconds`. */
export function timed(inner: Objective, seconds: number, failReason: string, showTimer = true): Objective {
  let elapsed = 0;
  const obj: Objective = {
    text: inner.text,
    blipKind: inner.blipKind,
    target: () => inner.target(),
    start(ctx) {
      elapsed = 0;
      inner.start(ctx);
    },
    update(ctx, dt) {
      elapsed += dt;
      const left = Math.max(0, Math.ceil(seconds - elapsed));
      obj.text = showTimer ? `${inner.text} ~y~${Math.floor(left / 60)}:${String(left % 60).padStart(2, '0')}~y~` : inner.text;
      const s = inner.update(ctx, dt);
      if (s !== 'running') return s;
      if (elapsed >= seconds) {
        ctx.fail(failReason);
        return 'failed';
      }
      return 'running';
    },
    skip: (ctx) => inner.skip(ctx),
    stop: (ctx) => inner.stop(ctx),
  };
  return obj;
}

/** Aim direction helper for scripted spawns: heading from `from` toward `to`. */
export { headingTo };
