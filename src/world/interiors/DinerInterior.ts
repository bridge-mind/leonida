import { Group } from 'three';
import type { StoreDef } from '../../data/shops';
import type { StoreInterior } from './ConvenienceInterior';
import { StoreKit, counter, entrance, storeOffset, type StoreLayout } from './StoreKit';

const ROOM = { w: 14, d: 10, h: 3.2 };

/** Waffle Hut: counter with stools along the west wall, booths along the east wall, checkerboard floor. */
export function createDinerInterior(store: StoreDef, slot: number): StoreInterior {
  const ox = storeOffset(slot);
  const P = (x: number, y: number, z: number) => ({ x: x + ox, y, z });

  const layout: StoreLayout = {
    counter: P(-2.5, 0, -0.5),
    clerk: { pos: P(-5.4, 0, -0.5), heading: Math.PI / 2 },
    register: P(-4.4, 1.1, -3.4),
    guards: [],
    bed: null,
    interactRadius: 1.9,
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
          floor: 0xe8e2d2,
          wall: 0xf6d97a,
          ceiling: 0xffffff,
          accent: 0xd94f3d,
        });

        // Checkerboard tiles on top of the floor.
        const dark = kit.std(0x2b2b30, 0.8);
        for (let i = -6; i < 7; i++) {
          for (let j = -4; j < 5; j++) {
            if ((i + j) % 2 !== 0) continue;
            kit.box(root, 1, 0.02, 1, dark, i + 0.5, 0, j + 0.5, {
              shadow: false,
            });
          }
        }

        // Counter runs north-south along the west side; register at the north end.
        layout.register = counter(kit, root, -4.4, -0.5, 6, Math.PI / 2, 0xd94f3d);
        const chrome = kit.std(0xcfd4dc, 0.3, 0.8);
        const seat = kit.std(0xd94f3d, 0.6);
        for (let i = 0; i < 5; i++) {
          const z = -2.7 + i * 1.1;
          kit.box(root, 0.08, 0.6, 0.08, chrome, -3.3, 0, z);
          kit.box(root, 0.42, 0.12, 0.42, seat, -3.3, 0.6, z, {
            collide: true,
          });
        }
        // Back bar: waffle irons, coffee pots, pie case.
        kit.box(root, 0.8, 1.0, 6, kit.std(0x9a3c30, 0.7), -6.5, 0, -0.5, {
          collide: true,
        });
        kit.box(root, 0.85, 0.05, 6, kit.std(0x2b2b33, 0.3, 0.4), -6.5, 1.0, -0.5, { shadow: false });
        for (let i = 0; i < 3; i++)
          kit.box(root, 0.45, 0.25, 0.45, kit.std(0x333940, 0.4, 0.6), -6.5, 1.05, -2.2 + i * 1.3, { shadow: false });
        kit.box(root, 0.5, 0.5, 1.2, kit.glow(0xfff1c2, 0.6), -6.5, 1.05, 1.9, {
          shadow: false,
        });

        // Booths along the east wall: table between two high-backed benches.
        const bench = kit.std(0x2f7ea8, 0.7);
        const table = kit.std(0xf1ece0, 0.5);
        for (let i = 0; i < 3; i++) {
          const z = -3 + i * 3;
          kit.box(root, 1.6, 0.45, 0.6, bench, 5.9, 0, z - 1.05, {
            collide: true,
          });
          kit.box(root, 1.6, 1.1, 0.12, bench, 5.9, 0, z - 1.4, {
            collide: true,
          });
          kit.box(root, 1.6, 0.45, 0.6, bench, 5.9, 0, z + 1.05, {
            collide: true,
          });
          kit.box(root, 1.6, 1.1, 0.12, bench, 5.9, 0, z + 1.4, {
            collide: true,
          });
          kit.box(root, 1.4, 0.06, 1.0, table, 5.9, 0.72, z, { collide: true });
          kit.box(root, 0.1, 0.72, 0.1, chrome, 5.9, 0, z);
          kit.box(root, 0.25, 0.3, 0.25, kit.glow(0xffb347, 0.5), 6.5, 0.78, z, { shadow: false });
        }

        kit.sign(root, 'WAFFLE HUT', 'OPEN 24 HOURS  •  BREAKFAST ALL DAY', '#ffd166', 4.6, 1.2, 0, 2.5, -4.9);
        kit.light(root, 0xfff0d0, 20, 13, -3, 3, -1);
        kit.light(root, 0xfff0d0, 20, 13, 4, 3, 1);

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
