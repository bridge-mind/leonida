/**
 * Performance counters. The renderer fills draw calls/triangles; systems set
 * their own entity counts each frame (`game.perf.counts.peds = n`).
 */
export class Perf {
  fps = 0;
  frameMs = 0;
  drawCalls = 0;
  triangles = 0;
  bodies = 0;
  colliders = 0;
  /** Entity counts by system, e.g. { peds: 80, vehicles: 25 }. */
  readonly counts: Record<string, number> = {};

  private frames = 0;
  private accum = 0;
  private lastSampleMs = 0;

  /** Called once per frame with the unscaled delta (seconds). */
  sample(realDt: number, frameMs: number): void {
    this.frames++;
    this.accum += realDt;
    this.frameMs = this.frameMs * 0.9 + frameMs * 0.1;
    if (this.accum >= 0.5) {
      this.fps = this.frames / this.accum;
      this.frames = 0;
      this.accum = 0;
      this.lastSampleMs = performance.now();
    }
  }

  set(name: string, value: number): void {
    this.counts[name] = value;
  }

  increment(name: string, by = 1): void {
    this.counts[name] = (this.counts[name] ?? 0) + by;
  }

  toJSON(): Record<string, unknown> {
    return {
      fps: Math.round(this.fps),
      frameMs: +this.frameMs.toFixed(2),
      drawCalls: this.drawCalls,
      triangles: this.triangles,
      bodies: this.bodies,
      colliders: this.colliders,
      counts: { ...this.counts },
      sampledAt: this.lastSampleMs,
    };
  }
}
