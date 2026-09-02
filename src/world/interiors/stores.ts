import type { StoreDef } from '../../data/shops';
import { createConvenienceInterior, type StoreInterior } from './ConvenienceInterior';
import { createDinerInterior } from './DinerInterior';
import { createGuardedInterior } from './GuardedInterior';
import { createGunStoreInterior } from './GunStoreInterior';
import { createSafehouseInterior } from './SafehouseInterior';

export type { StoreInterior } from './ConvenienceInterior';

/**
 * Picks the interior template for a store. `slot` must be unique per store
 * (1-based; slot 0 is the casino) so the interiors don't share collider space.
 */
export function createStoreInterior(store: StoreDef, slot: number): StoreInterior | null {
  switch (store.kind) {
    case 'gun':
      return createGunStoreInterior(store, slot);
    case 'convenience':
    case 'gas':
      return createConvenienceInterior(store, slot);
    case 'food':
      return createDinerInterior(store, slot);
    case 'safehouse':
      return createSafehouseInterior(store, slot);
    case 'jewelry':
    case 'bank':
      return createGuardedInterior(store, slot);
    case 'respray':
    case 'chopshop':
      return null;
  }
}
