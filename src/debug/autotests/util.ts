import type { Game } from '../../core/Game';

/**
 * Helpers for autotests. Each check prints `AUTOTEST PASS <suite>/<check>` or
 * `AUTOTEST FAIL <suite>/<check> <reason>`; the suite throws at the end if any check failed.
 */
export class Suite {
  private failures = 0;
  private checks = 0;

  constructor(
    readonly name: string,
    readonly game: Game,
  ) {}

  async check(label: string, fn: () => Promise<void> | void): Promise<void> {
    this.checks++;
    try {
      await fn();
      console.log(`AUTOTEST PASS ${this.name}/${label}`);
    } catch (err) {
      this.failures++;
      const reason = err instanceof Error ? err.message : String(err);
      console.log(`AUTOTEST FAIL ${this.name}/${label} ${reason}`);
    }
  }

  /** Throws if any check failed so the harness prints an overall FAIL. */
  finish(): void {
    if (this.failures > 0) throw new Error(`${this.failures}/${this.checks} checks failed`);
  }
}

export function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

export function assertClose(actual: number, expected: number, tolerance: number, label: string): void {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`${label}: expected ${expected} ± ${tolerance}, got ${actual}`);
  }
}

export function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Resolves after `n` render frames. */
export function waitFrames(game: Game, n: number): Promise<void> {
  const target = game.time.frame + n;
  return waitFor(() => game.time.frame >= target, 10000, 'frames');
}

export async function waitFor(predicate: () => boolean, timeoutMs: number, label: string): Promise<void> {
  const start = performance.now();
  while (!predicate()) {
    if (performance.now() - start > timeoutMs) throw new Error(`timeout waiting for ${label}`);
    await wait(16);
  }
}

export function horizontalDistance(a: { x: number; z: number }, b: { x: number; z: number }): number {
  const dx = a.x - b.x;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dz * dz);
}
