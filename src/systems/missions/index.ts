import type { V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { IMissionsPeer, MissionSummary } from '../../core/peers';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { MISSIONS } from '../../missions';
import type { MissionDef } from '../../missions/types';
import { RetryPrompt } from '../../ui/missions/RetryPrompt';
import { MissionsConfig } from './config';
import { ContactMarkers } from './Markers';
import { MissionRun, type RunStatus } from './Runner';

interface MissionsSave {
  completed: string[];
  active: string | null;
}

const PROMPT_KEY = 'mission';
const OBJECTIVE_BLIP = 'mission-objective';

/**
 * Story + job missions: contact markers, sequential objectives with GPS/HUD, pass/fail
 * cards, rewards, autosave and the retry prompt. Missions themselves live in
 * `src/missions/*`; this class only runs them.
 */
export class MissionSystem implements GameSystem, IMissionsPeer {
  private readonly defs = new Map<string, MissionDef>();
  private readonly done = new Set<string>();
  private run: MissionRun | null = null;
  private markers!: ContactMarkers;
  private retry: RetryPrompt | null = null;
  private readonly offs: (() => void)[] = [];
  private lastText: string | null = null;
  private readonly lastTarget = { x: NaN, y: 0, z: NaN };
  private hasBlip = false;
  private promptShown = false;
  /** Card seconds; the debug `complete` action shortens them. */
  private cardSeconds: number = MissionsConfig.cards.seconds;
  private unlockAllFlag = false;
  private newGameTimer = -1;
  /** Marker distance check runs on a short cadence; it is not frame-critical. */
  private markerTick = 0;
  private readonly ordered: MissionDef[] = [];

  constructor(private readonly game: Game) {
    for (const def of MISSIONS) {
      this.defs.set(def.id, def);
      this.ordered.push(def);
    }
  }

  init(): void {
    this.markers = new ContactMarkers(this.game.worldRoot, () => this.game.navigation);
    this.offs.push(
      this.game.events.on('ui:newGame', () => {
        if (this.done.size === 0 && !this.run) this.newGameTimer = MissionsConfig.newGameStartDelay;
      }),
      this.game.events.on('save:loaded', () => this.refreshMarkers()),
    );
    this.refreshMarkers();
  }

  // --- IMissionsPeer ---------------------------------------------------------------

  get active(): string | null {
    return this.run && this.run.status === 'running' ? this.run.def.id : null;
  }

  get completed(): readonly string[] {
    return Array.from(this.done);
  }

  available(): MissionSummary[] {
    const out: MissionSummary[] = [];
    for (const def of this.ordered) {
      if (!this.isAvailable(def)) continue;
      out.push({ id: def.id, name: def.name, contact: def.contact, position: def.position });
    }
    return out;
  }

  start(id: string): boolean {
    const def = this.defs.get(id);
    if (!def) {
      console.warn(`[missions] unknown mission '${id}'`);
      return false;
    }
    if (this.run && this.run.status === 'running') {
      this.game.ui?.notify('Finish the current mission first', 'warning');
      return false;
    }
    if (!this.isAvailable(def) && !this.unlockAllFlag) {
      this.game.ui?.notify(`${def.name} is locked`, 'warning');
      return false;
    }
    this.beginRun(def, false);
    return true;
  }

  abort(): void {
    const run = this.run;
    if (!run || run.status !== 'running') return;
    run.abort();
    this.endRun();
    this.game.ui?.notify(`${run.def.name} abandoned`, 'warning');
  }

  /** True while the running mission needs the companion (blocks character switching). */
  get companionRequired(): boolean {
    return this.run?.status === 'running' && (this.game.systems.get('companion') as { active?: boolean } | undefined)?.active === true;
  }

  get currentRun(): MissionRun | null {
    return this.run;
  }

  isCompleted(id: string): boolean {
    return this.done.has(id);
  }

  // --- Run lifecycle -------------------------------------------------------------------

  private isAvailable(def: MissionDef): boolean {
    if (this.done.has(def.id) && !def.repeatable) return false;
    if (def.unlockedAfter && !this.done.has(def.unlockedAfter) && !this.unlockAllFlag) return false;
    return true;
  }

  private beginRun(def: MissionDef, fast: boolean): void {
    this.retry?.choose('quit');
    this.markers.clear();
    this.hidePrompt();
    const run = new MissionRun(this.game, def, fast);
    this.run = run;
    this.lastText = null;
    this.game.audio.play('mission.start');
    this.game.events.emit('mission:started', { id: def.id });
    run.start();
    this.publish();
  }

  /** Clears HUD/GPS state after a run ends; markers return on the next refresh. */
  private endRun(): void {
    const game = this.game;
    game.ui?.setObjective(null);
    game.ui?.progressBar.hide();
    if (this.hasBlip) {
      game.navigation?.removeBlip(OBJECTIVE_BLIP);
      game.navigation?.setWaypoint(null);
      this.hasBlip = false;
    }
    this.lastTarget.x = NaN;
    this.lastText = null;
    this.refreshMarkers();
  }

  private async onPassed(run: MissionRun): Promise<void> {
    const game = this.game;
    this.done.add(run.def.id);
    this.endRun();
    const reward = Math.max(0, Math.round(run.reward));
    if (reward > 0) game.economy?.add(reward, 'mission');
    game.state.stats.missionsPassed = (game.state.stats.missionsPassed ?? 0) + 1;
    game.events.emit('mission:passed', { id: run.def.id, reward });
    game.save.save();
    await game.ui?.card('missionPassed', reward > 0 ? `+$${reward.toLocaleString()}` : 'Mission Passed', this.cardSeconds);
    this.refreshMarkers();
  }

  private async onFailed(run: MissionRun): Promise<void> {
    const game = this.game;
    const reason = run.failReason ?? 'failed';
    this.endRun();
    game.events.emit('mission:failed', { id: run.def.id, reason });
    if (reason === 'wasted' || reason === 'busted') {
      // Let the WASTED/BUSTED card and the respawn finish before stacking ours on top.
      await new Promise<void>((resolve) => {
        const off = game.events.on('player:respawned', () => {
          off();
          resolve();
        });
      });
      await this.delay(MissionsConfig.cards.afterRespawnDelay);
    }
    await game.ui?.card('missionFailed', reason, this.cardSeconds);
    if (this.run !== run) return;
    const choice = await this.retryPrompt().show(run.def.name, reason);
    if (choice === 'retry' && this.run === run) this.beginRun(run.def, run.fast);
  }

  private delay(seconds: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, seconds * 1000));
  }

  private retryPrompt(): RetryPrompt {
    if (!this.retry) this.retry = new RetryPrompt(this.game, document.getElementById('ui') ?? document.body);
    return this.retry;
  }

  // --- Loop ------------------------------------------------------------------------------

  fixedUpdate(dt: number): void {
    const run = this.run;
    if (!run || run.status !== 'running') return;
    run.update(dt);
    // `update` may have moved the run out of 'running'; the narrowing above does not know that.
    const status = run.status as RunStatus;
    if (status === 'passed') void this.onPassed(run);
    else if (status === 'failed') void this.onFailed(run);
    else this.publish();
  }

  update(): void {
    const realDt = this.game.time.realDt;
    this.markers.update(realDt);
    this.retry?.update();
    if (this.newGameTimer >= 0) {
      this.newGameTimer -= realDt;
      if (this.newGameTimer < 0 && this.done.size === 0 && !this.active) this.start('m1');
    }
    this.markerTick -= realDt;
    if (this.markerTick <= 0) {
      this.markerTick = 0.1;
      this.checkMarkers();
    }
    if (this.promptShown && this.game.input.wasPressed('KeyE') && !this.game.input.isBlocked && !(this.game.ui?.isModal ?? false)) {
      const id = this.game.systems.has('player') ? this.markers.at(this.game.player.position) : null;
      if (id) this.start(id);
    }
  }

  private checkMarkers(): void {
    if (this.active || !this.game.systems.has('player') || this.game.state.interior) {
      this.hidePrompt();
      return;
    }
    const id = this.markers.at(this.game.player.position);
    const def = id ? this.defs.get(id) : undefined;
    if (!def) {
      this.hidePrompt();
      return;
    }
    this.game.ui?.prompt(PROMPT_KEY, `E — Start ${def.name}`);
    this.promptShown = true;
  }

  private hidePrompt(): void {
    if (!this.promptShown) return;
    this.promptShown = false;
    this.game.ui?.clearPrompt(PROMPT_KEY);
  }

  /** Pushes objective text, waypoint and blip to the HUD when they change. */
  private publish(): void {
    const run = this.run;
    if (!run) return;
    const text = run.text;
    if (text !== this.lastText) {
      this.lastText = text;
      this.game.ui?.setObjective(text || null);
      if (text) this.game.events.emit('mission:objective', { id: run.def.id, text });
    }
    const target = run.target();
    const nav = this.game.navigation;
    if (!target) {
      if (this.hasBlip) {
        nav?.removeBlip(OBJECTIVE_BLIP);
        nav?.setWaypoint(null);
        this.hasBlip = false;
        this.lastTarget.x = NaN;
      }
      return;
    }
    const moved = Number.isNaN(this.lastTarget.x) || Math.hypot(target.x - this.lastTarget.x, target.z - this.lastTarget.z) > MissionsConfig.waypointMoveThreshold;
    if (!moved) return;
    this.lastTarget.x = target.x;
    this.lastTarget.y = target.y;
    this.lastTarget.z = target.z;
    if (this.hasBlip) nav?.removeBlip(OBJECTIVE_BLIP);
    nav?.addBlip({ id: OBJECTIVE_BLIP, kind: run.blipKind, position: { x: target.x, y: target.y, z: target.z } });
    nav?.setWaypoint(target);
    this.hasBlip = true;
  }

  private refreshMarkers(): void {
    if (!this.markers) return;
    const wanted = new Set<string>();
    if (!this.active) for (const def of this.ordered) if (this.isAvailable(def)) wanted.add(def.id);
    for (const def of this.ordered) {
      if (wanted.has(def.id) && !this.markers.has(def.id)) {
        const ground = this.game.physics.groundHeightAt(def.position.x, def.position.z) ?? def.position.y;
        this.markers.add(def.id, def.position, def.contact, ground);
      } else if (!wanted.has(def.id) && this.markers.has(def.id)) this.markers.remove(def.id);
    }
  }

  // --- Save ------------------------------------------------------------------------------

  serialize(): MissionsSave {
    return { completed: Array.from(this.done), active: this.active };
  }

  deserialize(data: unknown): void {
    const d = data as Partial<MissionsSave> | null;
    if (!d || !Array.isArray(d.completed)) return;
    if (this.run?.status === 'running') {
      this.run.abort();
      this.endRun();
    }
    this.done.clear();
    for (const id of d.completed) if (typeof id === 'string' && this.defs.has(id)) this.done.add(id);
    this.refreshMarkers();
    // An interrupted mission restarts from its first objective, once the world has settled.
    if (typeof d.active === 'string' && this.defs.has(d.active)) {
      const id = d.active;
      setTimeout(() => {
        if (!this.active) this.start(id);
      }, 0);
    }
  }

  dispose(): void {
    for (const off of this.offs) off();
    this.run?.abort();
    this.endRun();
    this.markers.dispose();
    this.retry?.dispose();
  }

  // --- Debug ------------------------------------------------------------------------------

  debugActions = {
    list: (): { id: string; name: string; state: 'done' | 'available' | 'locked' | 'active' }[] =>
      this.ordered.map((d) => ({
        id: d.id,
        name: d.name,
        state: this.active === d.id ? 'active' : this.done.has(d.id) && !d.repeatable ? 'done' : this.isAvailable(d) ? 'available' : 'locked',
      })),
    start: (id: unknown): boolean => this.start(String(id)),
    abort: (): void => this.abort(),
    unlockAll: (): void => {
      this.unlockAllFlag = true;
      this.refreshMarkers();
    },
    fastForward: (): string | null => {
      this.run?.fastForward();
      if (this.run?.status === 'running') this.publish();
      return this.run?.text ?? null;
    },
    /** Runs `id` to completion with fast-forward; resolves with the final status. */
    complete: async (id: unknown): Promise<string> => {
      const def = this.defs.get(String(id));
      if (!def) throw new Error(`unknown mission '${String(id)}'`);
      if (this.run?.status === 'running') this.abort();
      const prevSeconds = this.cardSeconds;
      this.cardSeconds = MissionsConfig.cards.fastSeconds;
      this.unlockAllFlag = true;
      this.beginRun(def, true);
      const run = this.run as MissionRun;
      try {
        for (let guard = 0; guard < 200 && run.status === 'running'; guard++) {
          await this.nextFrame();
          if (run.status === 'running') run.fastForward();
          await this.nextFrame();
        }
        if (run.status === 'passed') await this.delay(this.cardSeconds + 0.8);
        else if (run.status === 'failed') {
          // Dismiss our own retry prompt so a scripted run never leaves an input blocker behind.
          for (let i = 0; i < 100 && !(this.retry?.visible ?? false); i++) await this.delay(0.1);
          this.retry?.choose('quit');
        }
      } finally {
        this.cardSeconds = prevSeconds;
      }
      return run.status;
    },
    retry: (choice: unknown): void => this.retry?.choose(choice === 'quit' ? 'quit' : 'retry'),
    retryVisible: (): boolean => this.retry?.visible ?? false,
    objective: (): { text: string; index: number; count: number; target: V3 | null } | null =>
      this.run ? { text: this.run.text, index: this.run.objectiveIndex, count: this.run.objectiveCount, target: this.run.target() } : null,
    reset: (): void => {
      this.run?.abort();
      this.done.clear();
      this.endRun();
    },
  };

  private nextFrame(): Promise<void> {
    const target = this.game.time.frame + 1;
    return new Promise((resolve) => {
      const tick = (): void => {
        if (this.game.time.frame >= target) resolve();
        else setTimeout(tick, 8);
      };
      tick();
    });
  }
}

const module: SystemModule = {
  id: 'missions',
  priority: PRIORITY.missions,
  create: (game) => new MissionSystem(game),
};

export default module;
