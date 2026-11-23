import { Group, Vector3 } from 'three';
import { describe, expect, it } from 'vitest';

// Store signs paint a canvas. The collider assertions do not need a real DOM.
if (typeof document === 'undefined') {
  Object.assign(globalThis, {
    document: {
      createElement: () => ({
        width: 0,
        height: 0,
        getContext: () => null,
      }),
    },
  });
}
import { Config } from '../../src/core/Config';
import { STORE_BY_ID } from '../../src/data/shops';
import type { InteriorBuildContext } from '../../src/world/interiors/Interiors';
import { createGunStoreInterior } from '../../src/world/interiors/GunStoreInterior';
import { createSafehouseInterior } from '../../src/world/interiors/SafehouseInterior';
import { STORE_SLOT_SPACING } from '../../src/world/interiors/StoreKit';

interface Recorded {
  center: { x: number; y: number; z: number };
  half: { x: number; y: number; z: number };
}

/** Records every static box a store interior builds. */
function record(build: (ctx: InteriorBuildContext) => unknown): Recorded[] {
  const calls: Recorded[] = [];
  const origin = new Vector3(Config.interiors.origin.x, Config.interiors.origin.y, Config.interiors.origin.z);
  const ctx = {
    root: new Group(),
    physics: {
      addStaticCuboid(
        center: { x: number; y: number; z: number },
        half: { x: number; y: number; z: number },
      ) {
        calls.push({
          center: { x: center.x, y: center.y, z: center.z },
          half: { x: half.x, y: half.y, z: half.z },
        });
      },
    },
    origin,
    toWorld: (local: { x: number; y: number; z: number }) => new Vector3(local.x, local.y, local.z).add(origin),
    game: {},
  } as unknown as InteriorBuildContext;
  build(ctx);
  return calls;
}

describe('store interior colliders', () => {
  it('places the safehouse floor under the room, not a city-sized box at the origin', () => {
    const slot = 8;
    const interior = createSafehouseInterior(STORE_BY_ID.get('safehouse')!, slot);
    const calls = record(interior.definition.build);
    const floor = calls[0];
    expect(floor.center.x).toBeCloseTo(STORE_SLOT_SPACING * slot);
    expect(floor.center.y).toBeCloseTo(Config.interiors.origin.y - 0.05);
    expect(floor.half).toEqual({ x: 5.5, y: 0.05, z: 4.5 });
    for (const box of calls) {
      expect(box.half.x).toBeLessThan(20);
      expect(box.half.y).toBeLessThan(20);
      expect(box.half.z).toBeLessThan(20);
      expect(box.center.y).toBeGreaterThan(Config.interiors.origin.y - 1);
    }
  });

  it('places the gun store floor under its spawn', () => {
    const slot = 1;
    const interior = createGunStoreInterior(STORE_BY_ID.get('gunstore')!, slot);
    const floor = record(interior.definition.build)[0];
    expect(floor.center.x).toBeCloseTo(STORE_SLOT_SPACING * slot);
    expect(floor.center.y).toBeCloseTo(Config.interiors.origin.y - 0.05);
    expect(floor.half).toEqual({ x: 7, y: 0.05, z: 5 });
  });
});
