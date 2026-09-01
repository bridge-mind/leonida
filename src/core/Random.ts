/**
 * Seeded PRNG (mulberry32). World generation uses seeded instances so the city
 * is identical every run; gameplay uses `Random.unseeded()`.
 */
export class Random {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
    if (this.state === 0) this.state = 0x9e3779b9;
  }

  static fromString(text: string): Random {
    let h = 2166136261;
    for (let i = 0; i < text.length; i++) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return new Random(h >>> 0);
  }

  static unseeded(): Random {
    return new Random((Math.random() * 0xffffffff) >>> 0);
  }

  /** Uniform float in [0, 1). */
  next(): number {
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  range(min: number, max: number): number {
    return min + (max - min) * this.next();
  }

  /** Integer in [min, max] inclusive. */
  int(min: number, max: number): number {
    return min + Math.floor(this.next() * (max - min + 1));
  }

  chance(probability: number): boolean {
    return this.next() < probability;
  }

  pick<T>(items: readonly T[]): T {
    return items[Math.floor(this.next() * items.length)];
  }

  sign(): number {
    return this.next() < 0.5 ? -1 : 1;
  }

  /** Approximately normal (sum of three uniforms), mean 0, roughly ±1.5. */
  gaussian(): number {
    return this.next() + this.next() + this.next() - 1.5;
  }

  shuffle<T>(items: T[]): T[] {
    for (let i = items.length - 1; i > 0; i--) {
      const j = Math.floor(this.next() * (i + 1));
      const tmp = items[i];
      items[i] = items[j];
      items[j] = tmp;
    }
    return items;
  }

  /** Independent child generator, deterministic for a given label. */
  fork(label: string): Random {
    return Random.fromString(`${label}:${this.next()}`);
  }
}
