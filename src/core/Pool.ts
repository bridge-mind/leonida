/**
 * Simple object pool. `acquire()` returns a reset object; `release()` returns it
 * for reuse. Pools never shrink; warm them up at init so hot paths do not allocate.
 */
export class Pool<T> {
  private readonly free: T[] = [];
  private readonly live = new Set<T>();

  constructor(
    private readonly factory: () => T,
    private readonly reset?: (item: T) => void,
    warm = 0,
  ) {
    for (let i = 0; i < warm; i++) this.free.push(factory());
  }

  acquire(): T {
    const item = this.free.length > 0 ? (this.free.pop() as T) : this.factory();
    this.live.add(item);
    return item;
  }

  release(item: T): void {
    if (!this.live.delete(item)) return;
    this.reset?.(item);
    this.free.push(item);
  }

  releaseAll(): void {
    for (const item of this.live) {
      this.reset?.(item);
      this.free.push(item);
    }
    this.live.clear();
  }

  forEachLive(fn: (item: T) => void): void {
    for (const item of this.live) fn(item);
  }

  get activeCount(): number {
    return this.live.size;
  }

  get freeCount(): number {
    return this.free.length;
  }

  get size(): number {
    return this.live.size + this.free.length;
  }
}
