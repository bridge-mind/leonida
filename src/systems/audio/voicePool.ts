/**
 * Voice bookkeeping with hard caps. Pure logic (no Web Audio) so it is unit-testable.
 * When a list is full the quietest voice is stolen; ties go to the oldest.
 */
export interface PoolVoice {
  readonly loop: boolean;
  readonly startedAt: number;
  /** Estimated loudness at the listener, used to pick a steal victim. */
  volume: number;
  kill(): void;
}

export class VoicePool<T extends PoolVoice> {
  readonly oneShots: T[] = [];
  readonly loops: T[] = [];
  stolen = 0;

  constructor(
    readonly maxOneShots: number,
    readonly maxLoops: number,
  ) {}

  /** Registers a voice, stealing one first when the relevant cap is reached. */
  add(voice: T): void {
    const list = voice.loop ? this.loops : this.oneShots;
    const cap = voice.loop ? this.maxLoops : this.maxOneShots;
    while (list.length >= cap && list.length > 0) this.steal(list);
    list.push(voice);
  }

  /** Removes without killing. Idempotent. */
  remove(voice: T): boolean {
    const list = voice.loop ? this.loops : this.oneShots;
    const i = list.indexOf(voice);
    if (i < 0) return false;
    list.splice(i, 1);
    return true;
  }

  /** Kills and removes every voice for which `expired` is true. */
  prune(expired: (voice: T) => boolean): void {
    for (let i = this.oneShots.length - 1; i >= 0; i--) {
      const v = this.oneShots[i];
      if (expired(v)) {
        this.oneShots.splice(i, 1);
        v.kill();
      }
    }
    for (let i = this.loops.length - 1; i >= 0; i--) {
      const v = this.loops[i];
      if (expired(v)) {
        this.loops.splice(i, 1);
        v.kill();
      }
    }
  }

  killAll(): void {
    for (let i = this.oneShots.length - 1; i >= 0; i--) this.oneShots[i].kill();
    for (let i = this.loops.length - 1; i >= 0; i--) this.loops[i].kill();
    this.oneShots.length = 0;
    this.loops.length = 0;
  }

  get active(): number {
    return this.oneShots.length + this.loops.length;
  }

  private steal(list: T[]): void {
    let victim = 0;
    for (let i = 1; i < list.length; i++) {
      const a = list[i];
      const b = list[victim];
      if (a.volume < b.volume || (a.volume === b.volume && a.startedAt < b.startedAt)) victim = i;
    }
    const v = list[victim];
    list.splice(victim, 1);
    this.stolen++;
    v.kill();
  }
}
