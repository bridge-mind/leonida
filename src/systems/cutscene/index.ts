import type { V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { DialogueLine } from '../../core/peers';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { CutsceneConfig } from './config';

export type CutsceneStep =
  | { camera: { pos: V3; lookAt: V3; seconds?: number; fov?: number } }
  | { orbit: { center: V3; radius?: number; height?: number; speed?: number; seconds: number; fov?: number } }
  | { dialogue: DialogueLine[] }
  | { wait: number }
  /** Spawn/despawn hook; always runs, even when the scene is skipped or fast-forwarded. */
  | { run: (game: Game) => void };

export interface CutsceneDef {
  id: string;
  steps: CutsceneStep[];
  /** Default true; set false for quick radio-style dialogue without bars. */
  letterbox?: boolean;
}

interface Active {
  def: CutsceneDef;
  skipped: boolean;
  /** Scene has camera/orbit steps: only then is the camera taken over (radio chatter keeps the gameplay camera). */
  usesCamera: boolean;
  /** Wall-clock timer for `wait` and `orbit` steps (real seconds, so slow-mo cards do not stretch scenes). */
  timer: number;
  timerResolve: (() => void) | null;
  grace: number;
  resolve: () => void;
}

/**
 * Plays step lists: camera glides / orbits, dialogue, waits and spawn hooks. Blocks
 * player input, shows letterbox bars and restores the camera mode afterwards. Space
 * skips the rest of the scene (hooks still run so missions stay consistent).
 */
export class CutsceneSystem implements GameSystem {
  private active: Active | null = null;

  constructor(private readonly game: Game) {}

  init(): void {}

  get isPlaying(): boolean {
    return this.active !== null;
  }

  get currentId(): string | null {
    return this.active?.def.id ?? null;
  }

  /** Resolves when the scene ends or is skipped. A second `play` skips the first. */
  play(def: CutsceneDef, opts: { fast?: boolean } = {}): Promise<void> {
    if (opts.fast) {
      for (const step of def.steps) if ('run' in step) this.safeRun(step.run);
      return Promise.resolve();
    }
    if (this.active) {
      // Tear the previous scene down now: its own `finally` would find `this.active`
      // replaced and never pop the blocker or resolve its promise.
      const prev = this.active;
      this.skip();
      this.end(prev);
    }
    return new Promise<void>((resolve) => {
      const usesCamera = def.steps.some((s) => 'camera' in s || 'orbit' in s);
      const active: Active = { def, skipped: false, usesCamera, timer: 0, timerResolve: null, grace: CutsceneConfig.skipGrace, resolve };
      this.active = active;
      this.begin(active);
      void this.runSteps(active).finally(() => this.end(active));
    });
  }

  /** Ends the running scene at once; remaining hooks still execute. */
  skip(): void {
    const a = this.active;
    if (!a || a.skipped) return;
    a.skipped = true;
    a.timerResolve?.();
    a.timerResolve = null;
    this.game.ui?.dialogue.hide();
    this.releaseCamera(a);
  }

  private begin(active: Active): void {
    const game = this.game;
    game.input.pushBlocker(CutsceneConfig.inputBlocker);
    if (active.def.letterbox !== false) game.ui?.letterbox(true);
    game.ui?.clearPrompt('mission');
    if (active.usesCamera && game.systems.has('camera')) game.camera.setMode('cinematic');
  }

  private end(active: Active): void {
    if (this.active !== active) return;
    this.active = null;
    const game = this.game;
    game.ui?.dialogue.hide();
    if (active.def.letterbox !== false) game.ui?.letterbox(false);
    this.releaseCamera(active);
    game.input.popBlocker(CutsceneConfig.inputBlocker);
    active.resolve();
  }

  /** Hands the camera back once per scene; `stop()` restores the pre-cinematic mode and snaps. */
  private releaseCamera(active: Active): void {
    if (!active.usesCamera || !this.game.systems.has('camera')) return;
    active.usesCamera = false;
    this.game.camera.cinematic.stop();
  }

  private async runSteps(active: Active): Promise<void> {
    for (const step of active.def.steps) {
      if ('run' in step) {
        this.safeRun(step.run);
        continue;
      }
      if (active.skipped) continue;
      if ('camera' in step) await this.cameraStep(active, step.camera);
      else if ('orbit' in step) await this.orbitStep(active, step.orbit);
      else if ('dialogue' in step) await this.game.ui?.dialogue.show(step.dialogue);
      else if ('wait' in step) await this.waitStep(active, step.wait);
    }
  }

  private cameraStep(active: Active, c: { pos: V3; lookAt: V3; seconds?: number; fov?: number }): Promise<void> {
    if (!this.game.systems.has('camera')) return this.waitStep(active, c.seconds ?? CutsceneConfig.defaultMoveSeconds);
    return this.game.camera.cinematic.moveTo(c.pos, c.lookAt, c.seconds ?? CutsceneConfig.defaultMoveSeconds, c.fov);
  }

  private orbitStep(active: Active, o: { center: V3; radius?: number; height?: number; speed?: number; seconds: number; fov?: number }): Promise<void> {
    const O = CutsceneConfig.orbit;
    if (this.game.systems.has('camera')) {
      this.game.camera.cinematic.orbit(o.center, o.radius ?? O.radius, o.height ?? O.height, o.speed ?? O.speed, o.fov ?? O.fov);
    }
    return this.waitStep(active, o.seconds);
  }

  private waitStep(active: Active, seconds: number): Promise<void> {
    if (active.skipped) return Promise.resolve();
    return new Promise<void>((resolve) => {
      active.timer = seconds;
      active.timerResolve = resolve;
    });
  }

  private safeRun(fn: (game: Game) => void): void {
    try {
      fn(this.game);
    } catch (err) {
      console.error('[cutscene] hook failed', err);
    }
  }

  update(): void {
    const a = this.active;
    if (!a) return;
    const realDt = this.game.time.realDt;
    if (a.grace > 0) a.grace -= realDt;
    else if (this.game.input.wasPressed(CutsceneConfig.skipKey) && !(this.game.ui?.isModal ?? false)) {
      // The dialogue box owns Space while a line is up (UI advances it later this frame);
      // Space skips the scene only when no dialogue is showing.
      if (!(this.game.ui?.dialogue.active ?? false)) this.skip();
    }
    if (a.timerResolve) {
      a.timer -= realDt;
      if (a.timer <= 0) {
        const r = a.timerResolve;
        a.timerResolve = null;
        r();
      }
    }
  }

  dispose(): void {
    if (this.active) {
      this.skip();
      this.end(this.active);
    }
  }

  debugActions = {
    skip: (): void => this.skip(),
    playing: (): string | null => this.currentId,
    demo: (): Promise<void> => {
      const p = this.game.systems.has('player') ? this.game.player.position : { x: 0, y: 0, z: 0 };
      const center = { x: p.x, y: p.y + 1, z: p.z };
      return this.play({
        id: 'demo',
        steps: [
          { orbit: { center, seconds: 3 } },
          { dialogue: [{ speaker: 'Lucia', text: 'Cutscene demo.' }] },
          { camera: { pos: { x: p.x + 4, y: p.y + 2, z: p.z + 4 }, lookAt: center, seconds: 1.5 } },
        ],
      });
    },
  };
}

const module: SystemModule = {
  id: 'cutscene',
  priority: PRIORITY.cutscene,
  create: (game) => new CutsceneSystem(game),
};

export default module;
