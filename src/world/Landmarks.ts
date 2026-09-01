import type { V3 } from '../core/entities';
import type { BlipKind } from '../core/GameState';
import { LANDMARKS, type LandmarkDef } from '../data/landmarks';

/** Lookup helpers over the hand-placed landmarks. */
export class Landmarks {
  readonly list: readonly LandmarkDef[] = LANDMARKS;

  byId(id: string): LandmarkDef | undefined {
    return this.list.find((l) => l.id === id);
  }

  ofKind(kind: BlipKind): LandmarkDef[] {
    return this.list.filter((l) => l.kind === kind);
  }

  /** Closest landmark of a kind by horizontal distance to its door. */
  nearestOfKind(pos: V3, kind: BlipKind): LandmarkDef | null {
    let best: LandmarkDef | null = null;
    let bestD = Infinity;
    for (const l of this.list) {
      if (l.kind !== kind) continue;
      const dx = l.door.x - pos.x;
      const dz = l.door.z - pos.z;
      const d = dx * dx + dz * dz;
      if (d < bestD) {
        bestD = d;
        best = l;
      }
    }
    return best;
  }

  nearest(pos: V3): LandmarkDef {
    let best = this.list[0];
    let bestD = Infinity;
    for (const l of this.list) {
      const dx = l.door.x - pos.x;
      const dz = l.door.z - pos.z;
      const d = dx * dx + dz * dz;
      if (d < bestD) {
        bestD = d;
        best = l;
      }
    }
    return best;
  }

  /** Landmark whose footprint contains the point, if any. */
  at(x: number, z: number, margin = 0): LandmarkDef | null {
    for (const l of this.list) {
      const hw = l.footprint.w / 2 + margin;
      const hd = l.footprint.d / 2 + margin;
      if (Math.abs(x - l.position.x) <= hw && Math.abs(z - l.position.z) <= hd) return l;
    }
    return null;
  }
}
