import type { IVehicle, V3 } from '../../src/core/entities';
import { Events } from '../../src/core/Events';
import type { Game } from '../../src/core/Game';

/** Minimal player stand-in: position, mount, and the death hook the runner listens to. */
export class FakePlayer {
  position: V3 = { x: 0, y: 0, z: 0 };
  heading = 0;
  mountedVehicle: IVehicle | null = null;
  get isMounted(): boolean {
    return this.mountedVehicle !== null;
  }
  readonly entity = { kind: 'player' };
  teleport(pos: V3, heading?: number): void {
    this.position = { x: pos.x, y: pos.y, z: pos.z };
    if (heading !== undefined) this.heading = heading;
  }
  mount(v: IVehicle): void {
    this.mountedVehicle = v;
  }
}

export class FakeWanted {
  level = 0;
  cleared: string[] = [];
  setLevel(level: number): void {
    this.level = level;
  }
  clear(reason: string): void {
    this.level = 0;
    this.cleared.push(reason);
  }
}

export interface FakeGame {
  events: Events;
  player: FakePlayer;
  wanted: FakeWanted | undefined;
  casino: { chips: number } | undefined;
  economy: { cash: number } | undefined;
  ui: { notes: string[]; notify(text: string): void } | undefined;
  systems: { has(id: string): boolean; get(id: string): unknown };
  vehicles: { exitCurrent(): void } | undefined;
}

/**
 * Only the handles objectives actually touch: events, the player, optional peers.
 * `systems.has('player')` is what `playerPos` checks before reading `game.player`.
 */
export function makeGame(opts: { wanted?: boolean; casino?: boolean } = {}): { game: Game; fake: FakeGame } {
  const player = new FakePlayer();
  const fake: FakeGame = {
    events: new Events(),
    player,
    wanted: opts.wanted ? new FakeWanted() : undefined,
    casino: opts.casino ? { chips: 100 } : undefined,
    economy: opts.casino ? { cash: 50 } : undefined,
    ui: { notes: [], notify: (text) => fake.ui?.notes.push(text) },
    systems: { has: (id) => id === 'player', get: () => undefined },
    vehicles: {
      exitCurrent: () => {
        player.mountedVehicle = null;
      },
    },
  };
  return { game: fake as unknown as Game, fake };
}

export interface FakeVehicle {
  alive: boolean;
  wrecked: boolean;
  position: V3;
  heading: number;
  driverEntity: unknown;
}

/** Enough of IVehicle for goto/deliver/enterVehicle and the debug `forceMount` path. */
export function makeVehicle(pos: V3): IVehicle & FakeVehicle {
  const v = {
    alive: true,
    wrecked: false,
    position: { ...pos },
    heading: 0,
    driverEntity: null,
    get isAlive() {
      return v.alive;
    },
    get state() {
      return { wrecked: v.wrecked, exploded: false };
    },
    teleport(p: V3, h: number) {
      v.position.x = p.x;
      v.position.y = p.y;
      v.position.z = p.z;
      v.heading = h;
    },
    seatDoorPosition: (_seat: number, out: { set(x: number, y: number, z: number): unknown }) => {
      out.set(v.position.x + 1, v.position.y, v.position.z);
      return out;
    },
    setDriver() {},
    enter: () => 0,
  } as unknown as IVehicle & FakeVehicle;
  return v;
}
