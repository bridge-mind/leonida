import type { Game } from '../core/Game';

type ActionFn = (...args: unknown[]) => unknown;
type AutotestFn = (game: Game) => Promise<void>;
type AutotestLoader = () => Promise<{ default: AutotestFn }>;

/**
 * Debug harness: `game.debug.run('systemId.action', ...args)`, URL `&actions=`,
 * URL `&autotest=<name>`, F3 perf overlay, and console error counting.
 */
export class Debug {
  private readonly actions = new Map<string, Record<string, ActionFn>>();
  private readonly perfEl: HTMLElement | null;
  private perfVisible = false;
  private perfTimer = 0;
  /** Number of console.error calls and uncaught errors since boot. */
  errorCount = 0;
  warningCount = 0;
  readonly errors: string[] = [];
  autotestName: string | null = null;
  autotestResult: 'pending' | 'pass' | 'fail' | null = null;

  constructor(private readonly game: Game) {
    this.perfEl = document.getElementById('perf');
    this.hookConsole();
    window.addEventListener('keydown', (e) => {
      if (e.code === 'F3') {
        e.preventDefault();
        this.togglePerf();
      }
    });
    if (game.params.has('perf')) this.togglePerf(true);
    this.register('time', {
      set: (hour) => {
        game.time.setHour(Number(hour));
        return game.time.hour;
      },
      scale: (s) => {
        game.time.scale = Number(s);
        return game.time.scale;
      },
      clockScale: (s) => {
        game.time.clockScale = Number(s);
        return game.time.clockScale;
      },
      get: () => ({ hour: game.time.hour, day: game.time.day, label: game.time.clockLabel }),
    });
    this.register('core', {
      systems: () => game.systems.all.map((e) => ({ id: e.id, enabled: e.enabled, priority: e.module.priority })),
      perf: () => game.perf.toJSON(),
      save: () => game.save.save(),
      load: () => game.save.load(),
      errors: () => this.errors.slice(),
    });
  }

  private hookConsole(): void {
    const origError = console.error.bind(console);
    const origWarn = console.warn.bind(console);
    console.error = (...args: unknown[]) => {
      this.errorCount++;
      this.errors.push(args.map((a) => (a instanceof Error ? a.stack ?? a.message : String(a))).join(' '));
      origError(...args);
    };
    console.warn = (...args: unknown[]) => {
      this.warningCount++;
      origWarn(...args);
    };
    window.addEventListener('error', (e) => {
      this.errorCount++;
      this.errors.push(`uncaught: ${e.message}`);
    });
    window.addEventListener('unhandledrejection', (e) => {
      this.errorCount++;
      this.errors.push(`unhandled rejection: ${String(e.reason)}`);
    });
  }

  register(id: string, actions: Record<string, ActionFn>): void {
    const existing = this.actions.get(id);
    this.actions.set(id, existing ? { ...existing, ...actions } : { ...actions });
  }

  /** Lists `id.action` paths, for the console and docs. */
  list(): string[] {
    const out: string[] = [];
    for (const [id, acts] of this.actions) for (const name of Object.keys(acts)) out.push(`${id}.${name}`);
    return out.sort();
  }

  run(path: string, ...args: unknown[]): unknown {
    const dot = path.indexOf('.');
    if (dot < 0) throw new Error(`debug action '${path}' must be '<systemId>.<action>'`);
    const id = path.slice(0, dot);
    const name = path.slice(dot + 1);
    const acts = this.actions.get(id);
    const fn = acts?.[name];
    if (!fn) throw new Error(`unknown debug action '${path}' (known: ${this.list().join(', ')})`);
    return fn(...args);
  }

  /**
   * Parses `a.b:1:2,c.d` into calls. Args are JSON-parsed when possible, else strings.
   */
  runActionString(spec: string): unknown[] {
    const results: unknown[] = [];
    for (const item of spec.split(',')) {
      const trimmed = item.trim();
      if (!trimmed) continue;
      const [path, ...rawArgs] = trimmed.split(':');
      const args = rawArgs.map((a) => {
        try {
          return JSON.parse(a) as unknown;
        } catch {
          return a;
        }
      });
      try {
        results.push(this.run(path, ...args));
      } catch (err) {
        console.error(`[debug] action '${trimmed}' failed`, err);
      }
    }
    return results;
  }

  /** Runs URL `&actions=` then `&autotest=` after all systems are ready. */
  async afterReady(): Promise<void> {
    const actions = this.game.params.get('actions');
    if (actions) this.runActionString(actions);
    const autotest = this.game.params.get('autotest');
    if (autotest) await this.runAutotest(autotest);
  }

  async runAutotest(name: string): Promise<void> {
    this.autotestName = name;
    this.autotestResult = 'pending';
    // util.ts is a shared helper, not a suite; exclude it so `?autotest=util` reports "no such autotest".
    const loaders = import.meta.glob(['./autotests/*.ts', '!./autotests/util.ts']) as Record<string, AutotestLoader>;
    const entry = Object.entries(loaders).find(([path]) => path.endsWith(`/${name}.ts`));
    if (!entry) {
      this.autotestResult = 'fail';
      console.log(`AUTOTEST FAIL ${name} no such autotest (src/debug/autotests/${name}.ts)`);
      return;
    }
    try {
      const mod = await entry[1]();
      await mod.default(this.game);
      this.autotestResult = 'pass';
      console.log(`AUTOTEST PASS ${name}`);
    } catch (err) {
      this.autotestResult = 'fail';
      const reason = err instanceof Error ? err.message : String(err);
      console.log(`AUTOTEST FAIL ${name} ${reason}`);
      console.error(err);
    }
  }

  togglePerf(force?: boolean): void {
    this.perfVisible = force ?? !this.perfVisible;
    if (this.perfEl) this.perfEl.hidden = !this.perfVisible;
  }

  /** Called by Game once per frame after render. */
  update(): void {
    if (!this.perfVisible || !this.perfEl) return;
    const now = performance.now();
    if (now - this.perfTimer < 250) return;
    this.perfTimer = now;
    const p = this.game.perf;
    const counts = Object.entries(p.counts)
      .map(([k, v]) => `${k} ${v}`)
      .join('  ');
    this.perfEl.textContent =
      `${p.fps.toFixed(0)} fps  ${p.frameMs.toFixed(1)} ms\n` +
      `draw ${p.drawCalls}  tris ${(p.triangles / 1000).toFixed(0)}k\n` +
      `bodies ${p.bodies}  colliders ${p.colliders}\n` +
      `${counts}\n` +
      `clock ${this.game.time.clockLabel} day ${this.game.time.day}  x${this.game.time.scale.toFixed(2)}\n` +
      `errors ${this.errorCount}`;
  }
}
