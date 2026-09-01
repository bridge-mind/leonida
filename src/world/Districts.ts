import { DISTRICTS, rectContains, rectDistanceSq, type DistrictDef, type DistrictId } from '../data/districts';

/** Lookup helpers over the district rectangles. */
export class Districts {
  readonly list: readonly DistrictDef[] = DISTRICTS;

  byId(id: DistrictId): DistrictDef {
    const d = this.list.find((x) => x.id === id);
    if (!d) throw new Error(`unknown district ${id}`);
    return d;
  }

  /** District whose rectangle contains the point, or null (water, highway verges). */
  at(x: number, z: number): DistrictDef | null {
    for (const d of this.list) if (rectContains(d.rect, x, z)) return d;
    return null;
  }

  /** Nearest district by rectangle distance; never null. */
  nearest(x: number, z: number): DistrictDef {
    let best = this.list[0];
    let bestD = Infinity;
    for (const d of this.list) {
      const dist = rectDistanceSq(d.rect, x, z);
      if (dist < bestD) {
        bestD = dist;
        best = d;
      }
    }
    return best;
  }
}
