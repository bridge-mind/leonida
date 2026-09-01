/**
 * 2D (x/z) spatial hash for "who is within R" queries. Query results are written
 * into a caller-provided array so hot paths do not allocate.
 */
export class SpatialHash<T> {
  private readonly cells = new Map<number, T[]>();
  private readonly cellOf = new Map<T, number>();
  private readonly posX = new Map<T, number>();
  private readonly posZ = new Map<T, number>();

  constructor(readonly cellSize: number) {}

  private key(cx: number, cz: number): number {
    return (cx + 32768) * 65536 + (cz + 32768);
  }

  private coord(v: number): number {
    return Math.floor(v / this.cellSize);
  }

  insert(item: T, x: number, z: number): void {
    const k = this.key(this.coord(x), this.coord(z));
    let bucket = this.cells.get(k);
    if (!bucket) {
      bucket = [];
      this.cells.set(k, bucket);
    }
    bucket.push(item);
    this.cellOf.set(item, k);
    this.posX.set(item, x);
    this.posZ.set(item, z);
  }

  /** Move an item; cheap when it stays in the same cell. */
  update(item: T, x: number, z: number): void {
    const prev = this.cellOf.get(item);
    if (prev === undefined) {
      this.insert(item, x, z);
      return;
    }
    this.posX.set(item, x);
    this.posZ.set(item, z);
    const k = this.key(this.coord(x), this.coord(z));
    if (k === prev) return;
    this.removeFromBucket(item, prev);
    let bucket = this.cells.get(k);
    if (!bucket) {
      bucket = [];
      this.cells.set(k, bucket);
    }
    bucket.push(item);
    this.cellOf.set(item, k);
  }

  remove(item: T): void {
    const prev = this.cellOf.get(item);
    if (prev === undefined) return;
    this.removeFromBucket(item, prev);
    this.cellOf.delete(item);
    this.posX.delete(item);
    this.posZ.delete(item);
  }

  has(item: T): boolean {
    return this.cellOf.has(item);
  }

  private removeFromBucket(item: T, k: number): void {
    const bucket = this.cells.get(k);
    if (!bucket) return;
    const idx = bucket.indexOf(item);
    if (idx >= 0) {
      bucket[idx] = bucket[bucket.length - 1];
      bucket.pop();
    }
    if (bucket.length === 0) this.cells.delete(k);
  }

  /**
   * Collects every item within `radius` of (x, z) into `out` (cleared first).
   * Uses the stored positions for the exact distance test.
   */
  query(x: number, z: number, radius: number, out: T[]): T[] {
    out.length = 0;
    const r2 = radius * radius;
    const cx0 = this.coord(x - radius);
    const cx1 = this.coord(x + radius);
    const cz0 = this.coord(z - radius);
    const cz1 = this.coord(z + radius);
    for (let cx = cx0; cx <= cx1; cx++) {
      for (let cz = cz0; cz <= cz1; cz++) {
        const bucket = this.cells.get(this.key(cx, cz));
        if (!bucket) continue;
        for (let i = 0; i < bucket.length; i++) {
          const item = bucket[i];
          const dx = (this.posX.get(item) as number) - x;
          const dz = (this.posZ.get(item) as number) - z;
          if (dx * dx + dz * dz <= r2) out.push(item);
        }
      }
    }
    return out;
  }

  /** Items in the cells overlapping the axis-aligned rectangle (no exact test). */
  queryRect(x0: number, z0: number, x1: number, z1: number, out: T[]): T[] {
    out.length = 0;
    const cx0 = this.coord(Math.min(x0, x1));
    const cx1 = this.coord(Math.max(x0, x1));
    const cz0 = this.coord(Math.min(z0, z1));
    const cz1 = this.coord(Math.max(z0, z1));
    for (let cx = cx0; cx <= cx1; cx++) {
      for (let cz = cz0; cz <= cz1; cz++) {
        const bucket = this.cells.get(this.key(cx, cz));
        if (bucket) for (let i = 0; i < bucket.length; i++) out.push(bucket[i]);
      }
    }
    return out;
  }

  get count(): number {
    return this.cellOf.size;
  }

  clear(): void {
    this.cells.clear();
    this.cellOf.clear();
    this.posX.clear();
    this.posZ.clear();
  }
}
