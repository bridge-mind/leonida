import { Group } from 'three';
import type { StoreDef } from '../../data/shops';
import type { InteriorDefinition } from './Interiors';
import { StoreKit, counter, entrance, shelfRun, storeOffset, type StoreLayout } from './StoreKit';

export interface StoreInterior {
  definition: InteriorDefinition;
  layout: StoreLayout;
  /** Scene group of the built interior (set on first build); shop peds hang their proxy rigs here. */
  sceneRoot: Group | null;
}

const ROOM = { w: 12, d: 9, h: 3.2 };

/**
 * 24/7 convenience stores and the gas station share one floor plan: two shelf
 * aisles, fridges on the east wall, a counter with the register at the back.
 */
export function createConvenienceInterior(store: StoreDef, slot: number): StoreInterior {
  const ox = storeOffset(slot);
  const gas = store.kind === 'gas';
  const P = (x: number, y: number, z: number) => ({ x: x + ox, y, z });

  const layout: StoreLayout = {
    counter: P(-1, 0, -1.6),
    clerk: { pos: P(-1, 0, -3.5), heading: 0 },
    register: P(0.3, 1.1, -2.6),
    guards: [],
    bed: null,
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
          floor: gas ? 0x6d6b66 : 0xd9d6cf,
          wall: gas ? 0xc9c3b3 : 0xf0efe8,
          ceiling: 0xffffff,
          accent: gas ? 0xe0342b : 0x2fb1a6,
        });

        const goods = gas ? [0xf2c14e, 0xe0342b, 0x3e7cf3, 0xffffff] : [0xf25c54, 0x4cc9f0, 0xf9c74f, 0x90be6d, 0xffffff];
        layout.register = counter(kit, root, -1, -2.6, 4, 0, gas ? 0x5c3c2a : 0x2a4a5e);
        shelfRun(kit, root, -2.2, 1.2, 4.2, Math.PI / 2, goods);
        shelfRun(kit, root, 1.6, 1.2, 4.2, Math.PI / 2, goods);

        // Fridge wall: glowing glass doors on the east wall.
        for (let i = 0; i < 4; i++) {
          const z = -3 + i * 1.6;
          kit.box(root, 0.7, 2.1, 1.4, kit.std(0xe8ecef, 0.4, 0.4), 5.55, 0, z, { collide: true });
          kit.box(root, 0.05, 1.7, 1.1, kit.glow(gas ? 0xffa74a : 0x9be7ff, 0.6), 5.17, 0.25, z, { shadow: false });
        }
        // Hot-dog roller / snack stand next to the register.
        kit.box(root, 0.9, 0.35, 0.5, kit.std(0x444a52, 0.4, 0.6), 2.4, 1.05, -2.6, { shadow: false });
        kit.box(root, 1.2, 1.05, 0.6, kit.std(0x2a2a30, 0.6), 2.4, 0, -2.6, {
          collide: true,
        });
        kit.box(root, 0.8, 0.12, 0.3, kit.glow(gas ? 0xf2c14e : 0xff7a3d, 0.9), 2.4, 1.42, -2.6, { shadow: false });
        // Lotto / ATM box by the door.
        kit.box(root, 0.6, 1.6, 0.5, kit.std(0x22232b, 0.5, 0.3), -4.8, 0, 3.4, { collide: true });
        kit.box(root, 0.4, 0.5, 0.05, kit.glow(0x63f5b2, 0.8), -4.8, 0.9, 3.66, { shadow: false });

        kit.sign(
          root,
          gas ? 'XERO' : '24/7',
          gas ? 'GAS  •  SNACKS  •  ICE' : 'ALWAYS OPEN',
          gas ? '#ff5a4a' : '#63f5b2',
          3.4,
          1.1,
          -1,
          2.4,
          -4.4,
        );
        kit.light(root, 0xfff4e0, 18, 12, -2, 3, -1);
        kit.light(root, 0xfff4e0, 18, 12, 2.5, 3, 2);

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
