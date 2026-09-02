import type { V3 } from '../core/entities';
import type { Game } from '../core/Game';
import type { BlipKind } from '../core/GameState';

export type ObjectiveStatus = 'running' | 'done' | 'failed';

/**
 * One sequential step of a mission. Instances are created fresh per attempt by
 * `MissionDef.build`, so all per-attempt state lives on the object itself.
 */
export interface Objective {
  /** HUD line with `~y~noun~y~` markup; re-read every frame so it may change while running. */
  text: string;
  /** Kind of the HUD/minimap blip drawn at `target()`. */
  blipKind: BlipKind;
  /** GPS target for the waypoint and blip; null shows neither. */
  target(): V3 | null;
  start(ctx: MissionContext): void;
  update(ctx: MissionContext, dt: number): ObjectiveStatus;
  /** Debug fast-forward: force the objective into its done state (teleport, kill, clear...). */
  skip(ctx: MissionContext): void;
  /** Always called once after `start`, whatever the outcome. */
  stop(ctx: MissionContext): void;
}

/** Everything an objective or mission script needs; `game` is the only handle to the engine. */
export interface MissionContext {
  readonly game: Game;
  readonly missionId: string;
  /** Cash paid on pass; missions adjust it while running (bad debt, heat, gas money). */
  reward: number;
  /** Set by `fail`; the runner reads it when an objective reports 'failed'. */
  failReason: string | null;
  fail(reason: string): void;
  /** Registered functions run once when the mission passes, fails or is aborted (LIFO). */
  onCleanup(fn: () => void): void;
  /** True while the debug `complete` action drives the mission: cutscenes and waits end at once. */
  fast: boolean;
}

export interface MissionDef {
  id: string;
  name: string;
  contact: string;
  /** Contact marker (world, ground level). */
  position: V3;
  unlockedAfter?: string;
  reward: number;
  repeatable: boolean;
  /** Builds the objective list for one attempt; spawns may happen inside `action(...)` objectives. */
  build(ctx: MissionContext): Objective[];
}
