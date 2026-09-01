import {
  ACESFilmicToneMapping,
  Group,
  PerspectiveCamera,
  Scene,
  SRGBColorSpace,
  WebGLRenderer,
} from 'three';
import { Debug } from '../debug/Debug';
import { Physics } from '../physics/Physics';
import { NullAudio, type AudioApi } from '../systems/audio/api';
import type { CameraSystem } from '../systems/camera';
import type { EconomySystem } from '../systems/economy';
import type { InteriorsSystem } from '../systems/interiors';
import type { PlayerSystem } from '../systems/player';
import type { WorldSystem } from '../systems/world';
import { Config } from './Config';
import { Events } from './Events';
import { createGameState, type GameState } from './GameState';
import { Input } from './Input';
import { Perf } from './Perf';
import { Save } from './Save';
import { SystemRegistry, type GameSystem, type SystemEntry, type SystemModule } from './System';
import { Time } from './Time';

type ModuleLoader = () => Promise<{ default: SystemModule }>;

/**
 * Owns the render loop, the core services and the system registry.
 * Systems are discovered lazily from `src/systems/<id>/index.ts`.
 */
export class Game {
  readonly renderer: WebGLRenderer;
  readonly scene = new Scene();
  /** Root for the outdoor city; interiors hide it while inside. */
  readonly worldRoot = new Group();
  /** Root for interior scenes. */
  readonly interiorRoot = new Group();
  /** The Three.js camera the renderer uses. The camera system drives it. */
  readonly viewCamera: PerspectiveCamera;
  readonly events = new Events();
  readonly input: Input;
  readonly time: Time;
  readonly state: GameState = createGameState();
  readonly physics: Physics;
  readonly save: Save;
  readonly perf = new Perf();
  readonly debug: Debug;
  readonly systems = new SystemRegistry();
  readonly params: URLSearchParams;
  readonly nullAudio = new NullAudio();

  ready = false;
  private accumulator = 0;
  private lastFrameMs = 0;
  private running = false;
  private rafId = 0;

  constructor(readonly canvas: HTMLCanvasElement) {
    this.params = new URLSearchParams(window.location.search);
    this.renderer = new WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, Config.render.maxPixelRatio));
    this.renderer.setSize(window.innerWidth, window.innerHeight, false);
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = Config.render.toneMappingExposure;
    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.shadowMap.enabled = Config.render.shadows;
    this.renderer.info.autoReset = false;

    this.viewCamera = new PerspectiveCamera(
      Config.render.fov,
      window.innerWidth / window.innerHeight,
      Config.render.near,
      Config.render.far,
    );
    this.viewCamera.position.set(0, 3, 8);
    this.viewCamera.lookAt(0, 1, 0);

    this.scene.add(this.worldRoot, this.interiorRoot);
    this.input = new Input(canvas);
    this.time = new Time(this.events);
    this.physics = new Physics(this.events);
    this.save = new Save(this.events, this.time);
    this.debug = new Debug(this);

    window.addEventListener('resize', () => this.onResize());
  }

  // --- Typed accessors for foundation systems ------------------------------

  /** Throws if the world system is not loaded (it is part of every isolation set). */
  get world(): WorldSystem {
    return this.require<WorldSystem>('world');
  }

  get player(): PlayerSystem {
    return this.require<PlayerSystem>('player');
  }

  get camera(): CameraSystem {
    return this.require<CameraSystem>('camera');
  }

  /** Undefined when the economy system is not loaded (use optional chaining). */
  get economy(): EconomySystem | undefined {
    return this.systems.get<EconomySystem>('economy');
  }

  get interiors(): InteriorsSystem | undefined {
    return this.systems.get<InteriorsSystem>('interiors');
  }

  /** Always safe to call: falls back to a silent implementation when audio is not loaded. */
  get audio(): AudioApi {
    return (this.systems.get('audio') as unknown as AudioApi | undefined) ?? this.nullAudio;
  }

  private require<T extends GameSystem>(id: string): T {
    const sys = this.systems.get<T>(id);
    if (!sys) throw new Error(`System '${id}' is not loaded or has been disabled (check ?systems=)`);
    return sys;
  }

  // --- Lifecycle -----------------------------------------------------------

  /** Discovers, filters, sorts and initialises systems, then starts the loop. */
  async start(): Promise<void> {
    const loaders = import.meta.glob('../systems/*/index.ts') as Record<string, ModuleLoader>;
    const wanted = this.parseSystemsFilter();
    const modules: SystemModule[] = [];
    for (const [path, load] of Object.entries(loaders)) {
      const id = path.split('/').slice(-2, -1)[0];
      if (wanted && !wanted.has(id)) continue;
      try {
        const mod = await load();
        if (!mod.default || mod.default.id !== id) {
          console.error(`[system:${id}] index.ts must default-export a SystemModule with id '${id}'`);
          continue;
        }
        modules.push(mod.default);
      } catch (err) {
        console.error(`[system:${id}] failed to load module`, err);
      }
    }
    if (wanted) {
      for (const id of wanted) {
        if (!modules.some((m) => m.id === id)) console.warn(`[game] ?systems= requested unknown system '${id}'`);
      }
    }
    modules.sort((a, b) => a.priority - b.priority);

    for (const module of modules) {
      let entry: SystemEntry | undefined;
      try {
        const instance = module.create(this);
        entry = this.systems.add(module, instance);
        this.save.register(module.id, instance);
        if (instance.debugActions) this.debug.register(module.id, instance.debugActions);
        await instance.init();
      } catch (err) {
        if (entry) this.disable(entry, err);
        else console.error(`[system:${module.id}]`, err);
      }
    }

    this.ready = true;
    this.events.emit('game:ready', { systems: this.systems.ids });
    (window as unknown as { __GAME_READY__: boolean }).__GAME_READY__ = true;
    this.running = true;
    this.lastFrameMs = performance.now();
    this.rafId = requestAnimationFrame((t) => this.frame(t));
    await this.debug.afterReady();
  }

  private parseSystemsFilter(): Set<string> | null {
    const raw = this.params.get('systems');
    if (!raw) return null;
    const ids = raw
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
    return ids.length ? new Set(ids) : null;
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.rafId);
  }

  dispose(): void {
    this.stop();
    for (let i = this.systems.all.length - 1; i >= 0; i--) {
      const entry = this.systems.all[i];
      this.call(entry, () => entry.instance.dispose?.());
    }
    this.physics.dispose();
    this.renderer.dispose();
  }

  // --- Loop ----------------------------------------------------------------

  private frame(nowMs: number): void {
    if (!this.running) return;
    this.rafId = requestAnimationFrame((t) => this.frame(t));
    const frameStart = performance.now();
    const realDt = Math.min(Config.loop.maxFrameDt, Math.max(0, (nowMs - this.lastFrameMs) / 1000));
    this.lastFrameMs = nowMs;

    this.input.beginFrame();
    this.time.advance(realDt);
    const dt = this.time.dt;

    const fixed = Config.loop.fixedDt;
    this.accumulator += dt;
    let steps = 0;
    while (this.accumulator >= fixed && steps < Config.loop.maxSubsteps) {
      this.fixedStep(fixed);
      this.accumulator -= fixed;
      steps++;
    }
    // Drop backlog after a hitch instead of spiralling into more substeps.
    if (this.accumulator > fixed) this.accumulator = fixed;
    const alpha = this.accumulator / fixed;

    for (const entry of this.systems.all) {
      if (!entry.enabled || !entry.instance.update) continue;
      this.call(entry, () => entry.instance.update?.(dt, alpha));
    }

    this.state.timeOfDay = this.time.hour;
    this.state.day = this.time.day;

    this.renderer.info.reset();
    this.renderer.render(this.scene, this.viewCamera);
    this.perf.drawCalls = this.renderer.info.render.calls;
    this.perf.triangles = this.renderer.info.render.triangles;
    this.perf.bodies = this.physics.world.bodies.len();
    this.perf.colliders = this.physics.world.colliders.len();
    this.perf.sample(realDt, performance.now() - frameStart);
    this.debug.update();
    this.input.endFrame();
  }

  private fixedStep(dt: number): void {
    for (const entry of this.systems.all) {
      if (!entry.enabled || !entry.instance.fixedUpdate) continue;
      this.call(entry, () => entry.instance.fixedUpdate?.(dt));
    }
    this.physics.step(dt);
  }

  /** Runs `fn` for a system; on throw the system is disabled for the rest of the session. */
  call(entry: SystemEntry, fn: () => void): void {
    try {
      fn();
    } catch (err) {
      this.disable(entry, err);
    }
  }

  private disable(entry: SystemEntry, err: unknown): void {
    if (!entry.enabled) return;
    entry.enabled = false;
    entry.error = err;
    console.error(`[system:${entry.id}]`, err);
    this.events.emit('game:systemDisabled', { id: entry.id, error: err });
  }

  private onResize(): void {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.viewCamera.aspect = w / h;
    this.viewCamera.updateProjectionMatrix();
  }
}
