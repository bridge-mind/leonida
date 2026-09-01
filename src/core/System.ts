import type { Game } from './Game';

/**
 * A system module lives in `src/systems/<id>/index.ts` and default-exports one of these.
 * `id` must equal the folder name. Lower `priority` runs first (see PRIORITY bands).
 */
export interface SystemModule {
  id: string;
  priority: number;
  create(game: Game): GameSystem;
}

export interface GameSystem {
  init(): Promise<void> | void;
  /** 60 Hz, after inputs are applied, before `world.step()`. */
  fixedUpdate?(dt: number): void;
  /** Every render frame with interpolation alpha in [0, 1). */
  update?(dt: number, alpha: number): void;
  dispose?(): void;
  serialize?(): unknown;
  deserialize?(data: unknown): void;
  /** `game.debug.run('<id>.<action>', ...args)` dispatches here. */
  debugActions?: Record<string, (...args: unknown[]) => unknown>;
}

/** Priority bands from docs/TEAM-PLAN.md §3.1. */
export const PRIORITY = {
  core: 0,
  input: 10,
  time: 20,
  physics: 50,
  world: 100,
  player: 150,
  camera: 190,
  vehicles: 200,
  traffic: 210,
  peds: 220,
  pickups: 230,
  weapons: 300,
  damage: 310,
  particles: 320,
  wanted: 400,
  police: 410,
  economy: 500,
  shops: 510,
  robbery: 520,
  casino: 530,
  interiors: 540,
  missions: 600,
  cutscene: 610,
  companion: 620,
  characters: 630,
  navigation: 700,
  ui: 800,
  audio: 900,
} as const;

export interface SystemEntry {
  readonly id: string;
  readonly module: SystemModule;
  readonly instance: GameSystem;
  enabled: boolean;
  error: unknown;
}

/** Ordered registry of loaded systems with fault isolation. */
export class SystemRegistry {
  private readonly entries: SystemEntry[] = [];
  private readonly byId = new Map<string, SystemEntry>();

  add(module: SystemModule, instance: GameSystem): SystemEntry {
    const entry: SystemEntry = { id: module.id, module, instance, enabled: true, error: null };
    this.entries.push(entry);
    this.byId.set(module.id, entry);
    return entry;
  }

  /** Returns the live system instance, or undefined if it is not loaded or has been disabled. */
  get<T extends GameSystem = GameSystem>(id: string): T | undefined {
    const e = this.byId.get(id);
    return e && e.enabled ? (e.instance as T) : undefined;
  }

  has(id: string): boolean {
    return this.byId.get(id)?.enabled === true;
  }

  entry(id: string): SystemEntry | undefined {
    return this.byId.get(id);
  }

  get ids(): string[] {
    return this.entries.map((e) => e.id);
  }

  get all(): readonly SystemEntry[] {
    return this.entries;
  }
}
