import { Group } from 'three';
import type { StoreDef } from '../../data/shops';
import type { StoreInterior } from './ConvenienceInterior';
import { StoreKit, counter, entrance, storeOffset, type StoreLayout } from './StoreKit';

const ROOM = { w: 16, d: 12, h: 4 };

/**
 * Jewelry store and bank shells: enterable, staffed and guarded by two armed
 * guards flanking the entrance. Robbing them is mission content; here they only
 * need to look the part and react to drawn weapons.
 */
export function createGuardedInterior(store: StoreDef, slot: number): StoreInterior {
  const ox = storeOffset(slot);
  const bank = store.kind === 'bank';
  const P = (x: number, y: number, z: number) => ({ x: x + ox, y, z });

  const layout: StoreLayout = {
    counter: P(0, 0, -2.4),
    clerk: { pos: P(0, 0, -4.6), heading: 0 },
    register: P(2.3, 1.1, -3.6),
    guards: [
      { pos: P(-6, 0, 3.6), heading: -Math.PI * 0.75 },
      { pos: P(6, 0, 3.6), heading: Math.PI * 0.75 },
    ],
    bed: null,
    interactRadius: 2,
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
          floor: bank ? 0xcfc7b8 : 0x1c1c22,
          wall: bank ? 0xe4e9ee : 0x2a2733,
          ceiling: 0xf4f4f4,
          accent: bank ? 0x2c8c5a : 0xd4af37,
        });

        const gold = kit.std(0xd4af37, 0.3, 0.9);
        if (bank) {
          // Teller counter with bars, rope line, vault door in the back wall.
          layout.register = counter(kit, root, 0, -3.6, 8, 0, 0x2c8c5a);
          for (let i = 0; i < 9; i++) kit.box(root, 0.04, 1.4, 0.04, kit.std(0x8a8f96, 0.3, 0.8), -4 + i, 1.1, -3.6, { shadow: false });
          kit.box(root, 8.2, 0.1, 0.1, kit.std(0x8a8f96, 0.3, 0.8), 0, 2.5, -3.6, { shadow: false });
          kit.box(root, 3.2, 3.2, 0.4, kit.std(0x55606a, 0.3, 0.9), 5.8, 0, -5.75, { collide: true });
          kit.box(root, 0.9, 0.9, 0.2, kit.std(0x8a8f96, 0.2, 1), 5.8, 1.2, -5.5, { shadow: false });
          for (let i = 0; i < 4; i++) {
            kit.box(root, 0.08, 1.0, 0.08, gold, -3 + i * 2, 0, 1.2, {
              collide: true,
            });
            if (i < 3) kit.box(root, 2, 0.05, 0.05, kit.std(0x8b1e2d, 0.9), -2 + i * 2, 0.9, 1.2, { shadow: false });
          }
          for (let i = 0; i < 3; i++) kit.box(root, 0.6, 1.2, 0.5, kit.std(0x33393f, 0.5, 0.3), -6.5 + i * 0.8, 0, -1.5, { collide: true });
          kit.sign(root, 'FLEECA', 'SAVINGS  •  LOANS  •  SECURITY', '#2fd47a', 4.6, 1.3, 0, 3.0, -5.85);
          kit.light(root, 0xfff7e6, 26, 16, -3, 3.8, -1);
          kit.light(root, 0xfff7e6, 26, 16, 3, 3.8, 2);
        } else {
          // Glass display cases in a ring, velvet floor, chandelier glow.
          layout.register = counter(kit, root, 0, -3.6, 5, 0, 0x3a2f4a);
          const base = kit.std(0x14121a, 0.5, 0.3);
          const glass = kit.std(0xbfe6ff, 0.05, 0.7);
          const cases: Array<[number, number, number]> = [
            [-4.5, -0.5, Math.PI / 2],
            [4.5, -0.5, Math.PI / 2],
            [-2.3, 1.6, 0],
            [2.3, 1.6, 0],
          ];
          for (const [x, z, rotY] of cases) {
            kit.box(root, 3, 0.9, 0.8, base, x, 0, z, { collide: true, rotY });
            kit.box(root, 2.9, 0.5, 0.75, glass, x, 0.9, z, {
              rotY,
              shadow: false,
            });
            for (let i = 0; i < 4; i++) {
              const t = -1.1 + i * 0.73;
              kit.box(root, 0.18, 0.12, 0.18, gold, x + Math.cos(rotY) * t, 0.95, z - Math.sin(rotY) * t, { shadow: false });
            }
          }
          kit.box(root, 1.2, 0.5, 1.2, kit.glow(0xfff2c2, 1.2), 0, 3.4, -0.5, {
            shadow: false,
          });
          kit.box(root, 6, 0.02, 4, kit.std(0x5a1e2e, 1), 0, 0, -0.5, {
            shadow: false,
          });
          kit.sign(root, 'VANGELICO', 'FINE JEWELLERY SINCE 1925', '#ffd66b', 4.4, 1.2, 0, 3.0, -5.85);
          kit.light(root, 0xffefc9, 24, 15, 0, 3.6, -0.5);
          kit.light(root, 0xffefc9, 14, 12, 0, 3.6, 3.5);
        }

        // Guard posts: small podiums so the posts read even before the peds spawn.
        for (const g of layout.guards) {
          kit.box(root, 0.9, 0.05, 0.9, kit.std(0x1a1a20, 0.6), g.pos.x - ox, 0, g.pos.z, { shadow: false });
        }

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
