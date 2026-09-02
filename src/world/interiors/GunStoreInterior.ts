import { Group } from 'three';
import type { StoreDef } from '../../data/shops';
import type { StoreInterior } from './ConvenienceInterior';
import { StoreKit, counter, entrance, storeOffset, type StoreLayout } from './StoreKit';

const ROOM = { w: 14, d: 10, h: 3.4 };

/** Weapon silhouettes as stacked boxes: [width, height, depth, dy] per part, hung on a wall rack. */
const WEAPON_SHAPES: Record<'pistol' | 'smg' | 'shotgun' | 'rifle' | 'sniper', Array<[number, number, number, number]>> = {
  pistol: [
    [0.28, 0.07, 0.05, 0.12],
    [0.08, 0.14, 0.05, 0],
  ],
  smg: [
    [0.55, 0.09, 0.06, 0.14],
    [0.08, 0.18, 0.06, 0],
    [0.06, 0.16, 0.06, -0.02],
  ],
  shotgun: [
    [0.95, 0.07, 0.06, 0.1],
    [0.3, 0.12, 0.06, 0.02],
  ],
  rifle: [
    [0.85, 0.08, 0.06, 0.14],
    [0.06, 0.2, 0.06, -0.02],
    [0.3, 0.12, 0.06, 0.02],
  ],
  sniper: [
    [1.2, 0.07, 0.06, 0.14],
    [0.3, 0.06, 0.06, 0.25],
    [0.34, 0.12, 0.06, 0.02],
  ],
};

/**
 * Bullet Barn: a long counter with the clerk, wall racks of weapon silhouettes
 * behind him, ammo crates and an armour mannequin on the side wall.
 */
export function createGunStoreInterior(store: StoreDef, slot: number): StoreInterior {
  const ox = storeOffset(slot);
  const P = (x: number, y: number, z: number) => ({ x: x + ox, y, z });

  const layout: StoreLayout = {
    counter: P(0, 0, -1.2),
    clerk: { pos: P(0, 0, -3.9), heading: 0 },
    register: P(1.8, 1.1, -3),
    guards: [],
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
          floor: 0x4a3b2e,
          wall: 0x8c7b66,
          ceiling: 0xd8d0c4,
          accent: 0xd08a2e,
        });

        layout.register = counter(kit, root, 0, -3, 5, 0, 0x5a3b22);
        // Glass display case in front of the counter with pistols under it.
        kit.box(root, 5, 0.9, 0.7, kit.std(0x2e2a26, 0.5), 0, 0, -1.2 - 0.9, {
          collide: true,
        });
        kit.box(root, 4.9, 0.05, 0.65, kit.std(0x9ad7ff, 0.1, 0.6), 0, 0.9, -2.1, { shadow: false });
        for (let i = 0; i < 5; i++) weaponShape(kit, root, 'pistol', -2 + i, 0.6, -2.1, 0);

        // Wall racks behind the counter.
        const rackMat = kit.std(0x2b2018, 0.8);
        kit.box(root, 10, 2.2, 0.08, rackMat, 0, 0.6, -4.9);
        const hung: Array<keyof typeof WEAPON_SHAPES> = ['rifle', 'sniper', 'shotgun', 'smg', 'rifle', 'shotgun', 'sniper'];
        hung.forEach((id, i) => weaponShape(kit, root, id, -4.2 + i * 1.4, 1.9, -4.82, 0));
        hung.forEach((id, i) => weaponShape(kit, root, id, -4.2 + i * 1.4, 1.1, -4.82, 0));

        // Side wall rack (west) with long guns, ammo crates stacked below.
        kit.box(root, 0.08, 2, 5, rackMat, -6.9, 0.8, -1);
        for (let i = 0; i < 4; i++)
          weaponShape(kit, root, i % 2 ? 'rifle' : 'shotgun', -6.82, 1.6 + (i % 2) * 0.6, -3 + i * 1.2, Math.PI / 2);
        const crate = kit.std(0x6d7a3a, 0.9);
        for (let i = 0; i < 6; i++) {
          kit.box(root, 0.6, 0.4, 0.4, crate, -6.3 + (i % 2) * 0.65, Math.floor(i / 2) * 0.42, 1.8, { collide: i < 2 });
        }
        // Armour mannequin on the east wall.
        kit.box(root, 0.5, 1.75, 0.5, kit.std(0x3c3c44, 0.7), 6.2, 0, -1, {
          collide: true,
        });
        kit.box(root, 0.56, 0.7, 0.56, kit.std(0x1e2a44, 0.5, 0.2), 6.2, 0.85, -1, { shadow: false });
        kit.box(root, 0.4, 0.6, 0.05, kit.glow(0x63f5b2, 0.6), 6.68, 0.9, -1, {
          rotY: Math.PI / 2,
          shadow: false,
        });

        // Target poster and neon.
        kit.sign(root, 'BULLET BARN', 'GUNS  •  AMMO  •  ARMOUR', '#ffb347', 4.4, 1.2, 0, 2.6, -4.85);
        kit.sign(root, 'NO REFUNDS', 'ASK ABOUT OUR LOYALTY CARD', '#ff5a4a', 2.2, 0.8, 5.5, 2.3, -4.85);
        kit.light(root, 0xffe1b0, 22, 14, -2.5, 3.2, -1);
        kit.light(root, 0xffe1b0, 22, 14, 2.5, 3.2, 2);

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

function weaponShape(kit: StoreKit, root: Group, id: keyof typeof WEAPON_SHAPES, x: number, y: number, z: number, rotY: number): void {
  const mat = kit.std(0x15151a, 0.4, 0.7);
  const shape = WEAPON_SHAPES[id];
  const along = Math.cos(rotY);
  const across = Math.sin(rotY);
  for (const [w, h, d, dy] of shape) {
    // Stock/grip parts hang below the barrel line; shift grips back along the barrel.
    const back = (shape[0][0] - w) * 0.3;
    kit.box(root, w, h, d, mat, x - back * along, y + dy, z + back * across, {
      rotY,
      shadow: false,
    });
  }
}
