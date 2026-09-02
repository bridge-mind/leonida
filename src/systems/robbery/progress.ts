/**
 * Pure timing model for a hold-up: a fixed-duration bag fill that gunshots
 * shorten. Kept free of engine imports so it can be unit-tested.
 */
export class RobberyClock {
  private elapsed = 0;
  private remaining: number;
  shots = 0;

  constructor(
    readonly duration: number,
    private readonly shotCut: number,
    private readonly maxShots: number,
  ) {
    this.remaining = duration;
  }

  /** Advances time; returns true once the bag is full. */
  tick(dt: number): boolean {
    if (dt > 0) {
      this.elapsed += dt;
      this.remaining = Math.max(0, this.remaining - dt);
    }
    return this.done;
  }

  /** A gunshot scares the clerk into working faster; only the first `maxShots` count. */
  onShot(): boolean {
    if (this.shots >= this.maxShots) return false;
    this.shots++;
    this.remaining *= 1 - this.shotCut;
    return true;
  }

  get done(): boolean {
    return this.remaining <= 0;
  }

  get secondsLeft(): number {
    return this.remaining;
  }

  /** 0..1 fill fraction for the progress bar. */
  get fraction(): number {
    const total = this.elapsed + this.remaining;
    return total <= 0 ? 1 : Math.min(1, this.elapsed / total);
  }
}
