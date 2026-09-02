import { Group } from 'three';
import type { StoreDef } from '../../data/shops';
import type { StoreInterior } from './ConvenienceInterior';
import { StoreKit, entrance, storeOffset, type StoreLayout } from './StoreKit';

const ROOM = { w: 11, d: 9, h: 3 };

/** Safehouse apartment: bed (save point), wardrobe, couch + TV, kitchenette. */
export function createSafehouseInterior(store: StoreDef, slot: number): StoreInterior {
  const ox = storeOffset(slot);
  const P = (x: number, y: number, z: number) => ({ x: x + ox, y, z });

  const layout: StoreLayout = {
    counter: P(3.2, 0, -1.5),
    clerk: null,
    register: P(3.2, 1.2, -3),
    guards: [],
    bed: P(-3.1, 0, -1.2),
    interactRadius: 1.8,
  };

  const result: StoreInterior = {
    layout,
    sceneRoot: null,
    definition: {
      id: store.id,
      ...entrance(ox, ROOM.d),
      build(ctx) {
        const kit = new StoreKit(ctx, ox);
        const root = new Group();
        ctx.root.add(root);
        result.sceneRoot = root;
        kit.room(root, {
          w: ROOM.w,
          d: ROOM.d,
          h: ROOM.h,
          floor: 0x7a5a3e,
          wall: 0xe9dfd0,
          ceiling: 0xf7f3ec,
          accent: 0x63f5b2,
        });

        // Bed against the west wall.
        kit.box(root, 2.0, 0.45, 2.6, kit.std(0x3b2e22, 0.8), -4.3, 0, -2.6, {
          collide: true,
        });
        kit.box(root, 1.9, 0.25, 2.4, kit.std(0xf2f0ea, 0.9), -4.3, 0.45, -2.6, { shadow: false });
        kit.box(root, 1.9, 0.12, 1.4, kit.std(0x2f6f9f, 0.9), -4.3, 0.7, -2.2, {
          shadow: false,
        });
        kit.box(root, 0.7, 0.15, 0.45, kit.std(0xffffff, 0.9), -4.3, 0.7, -3.6, { shadow: false });
        kit.box(root, 2.1, 1.1, 0.12, kit.std(0x3b2e22, 0.8), -4.3, 0, -3.95);
        // Nightstand with lamp.
        kit.box(root, 0.5, 0.55, 0.5, kit.std(0x3b2e22, 0.8), -2.9, 0, -3.6, {
          collide: true,
        });
        kit.box(root, 0.25, 0.3, 0.25, kit.glow(0xffe3a8, 1), -2.9, 0.55, -3.6, { shadow: false });
        // Wardrobe next to the bed.
        kit.box(root, 1.4, 2.2, 0.7, kit.std(0x5a4030, 0.8), -1.4, 0, -4.05, {
          collide: true,
        });
        kit.box(root, 0.04, 1.6, 0.05, kit.std(0xcfd4dc, 0.3, 0.8), -1.4, 0.4, -3.68, { shadow: false });

        // Couch facing a TV on the east wall.
        kit.box(root, 0.9, 0.5, 2.4, kit.std(0x2c3d5a, 0.9), 1.6, 0, 0.5, {
          collide: true,
        });
        kit.box(root, 0.3, 0.9, 2.4, kit.std(0x2c3d5a, 0.9), 1.15, 0, 0.5, {
          collide: true,
        });
        kit.box(root, 0.8, 0.5, 0.8, kit.std(0x2b2b30, 0.6), 3.2, 0, 0.5, {
          collide: true,
        });
        kit.box(root, 0.08, 1.0, 1.7, kit.glow(0x4aa3ff, 0.7), 5.3, 0.7, 0.5, {
          shadow: false,
        });
        kit.box(root, 1.6, 0.6, 0.5, kit.std(0x2b2b30, 0.6), 5.2, 0, 0.5, {
          collide: true,
          rotY: Math.PI / 2,
        });

        // Kitchenette on the north-east corner.
        kit.box(root, 3.6, 0.95, 0.65, kit.std(0x8d99a6, 0.5, 0.2), 3.2, 0, -4.05, { collide: true });
        kit.box(root, 3.7, 0.05, 0.7, kit.std(0x2b2b33, 0.3, 0.4), 3.2, 0.95, -4.05, { shadow: false });
        kit.box(root, 0.5, 0.15, 0.4, kit.std(0xcfd4dc, 0.3, 0.8), 2.2, 1.0, -4.05, { shadow: false });
        kit.box(root, 0.8, 1.9, 0.7, kit.std(0xdfe4e8, 0.4, 0.5), 5.0, 0, -4.05, { collide: true });
        // Rug and a couple of bottles on the counter.
        kit.box(root, 3, 0.02, 2.2, kit.std(0xa84a3a, 1), 0.2, 0, 0.6, {
          shadow: false,
        });
        kit.box(root, 0.12, 0.3, 0.12, kit.std(0x3a8f4a, 0.3), 3.8, 1.0, -4.0, {
          shadow: false,
        });
        kit.box(root, 0.12, 0.25, 0.12, kit.std(0xd9a441, 0.3), 4.1, 1.0, -4.1, { shadow: false });

        kit.sign(root, 'HOME', 'PRESS E AT THE BED TO SAVE', '#63f5b2', 2.2, 0.7, -4.3, 2.2, -4.4);
        kit.light(root, 0xffe8c8, 14, 11, -2.5, 2.7, -1.5);
        kit.light(root, 0xffe8c8, 14, 11, 2.5, 2.7, 1);

        return {
          root,
          dispose() {
            root.removeFromParent();
            kit.dispose();
          },
        };
      },
    },
  };

  return result;
}
