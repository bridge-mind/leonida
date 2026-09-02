import type { Game } from '../../core/Game';
import { ARMOR_AMOUNT, ARMOR_PRICE, BUYABLE_WEAPONS, FOOD, WEAPON_PRICES, type BuyableWeaponId, type FoodId } from '../../data/prices';
import { WEAPONS } from '../../data/weapons';
import type { CatalogueRow, PurchaseResult } from '../../ui/shops/GunStore';

/**
 * Everything the player can buy over a counter: weapons (one clip included),
 * ammo for owned weapons, body armour and food. Pure game logic; the gun-store
 * panel and the debug actions both go through here.
 */
export class Purchases {
  constructor(private readonly game: Game) {}

  /** Gun-store catalogue as the panel displays it. */
  rows(): CatalogueRow[] {
    const weapons = this.game.weapons;
    const cash = this.game.economy?.cash ?? 0;
    const rows: CatalogueRow[] = BUYABLE_WEAPONS.map((id: BuyableWeaponId) => {
      const def = WEAPONS[id];
      const owned = weapons?.has(id) ?? false;
      const price = owned ? WEAPON_PRICES[id].ammo : WEAPON_PRICES[id].price;
      const ammo = owned ? (weapons?.inventory()[id] ?? 0) : 0;
      return {
        id,
        name: def.name,
        detail: owned ? `${def.magazine}-round clip · ${ammo} in stock` : `${def.damage} dmg · ${def.magazine}-round clip`,
        action: owned ? 'Buy ammo' : 'Buy',
        price,
        affordable: cash >= price,
        disabled: !weapons,
      };
    });
    const hasPlayer = this.game.systems.has('player');
    const armor = hasPlayer ? this.game.player.armor : 0;
    rows.push({
      id: 'armor',
      name: 'Body Armour',
      detail: `${Math.round(armor)}/${ARMOR_AMOUNT} worn`,
      action: armor >= ARMOR_AMOUNT ? 'Full' : 'Buy',
      price: ARMOR_PRICE,
      affordable: cash >= ARMOR_PRICE,
      disabled: armor >= ARMOR_AMOUNT || !hasPlayer,
    });
    return rows;
  }

  buy(item: string): PurchaseResult {
    const economy = this.game.economy;
    if (!economy) return { ok: false, message: 'No wallet.' };
    if (item === 'armor') {
      if (!this.game.systems.has('player')) return { ok: false, message: 'Nobody to wear it.' };
      const player = this.game.player;
      if (player.armor >= ARMOR_AMOUNT) return { ok: false, message: 'Already wearing full armour.' };
      if (!economy.trySpend(ARMOR_PRICE, 'shop:armor')) return this.denied(ARMOR_PRICE);
      player.addArmor(ARMOR_AMOUNT);
      return this.bought('armor', ARMOR_PRICE, 'Body armour fitted.');
    }
    if (!isBuyable(item)) return { ok: false, message: 'Not stocked.' };
    const weaponId: BuyableWeaponId = item;
    const weapons = this.game.weapons;
    if (!weapons) return { ok: false, message: 'No weapons system.' };
    const def = WEAPONS[weaponId];
    if (weapons.has(weaponId)) {
      const price = WEAPON_PRICES[weaponId].ammo;
      if (!economy.trySpend(price, 'shop:ammo')) return this.denied(price);
      weapons.addAmmo(weaponId, def.magazine);
      return this.bought(`${weaponId}:ammo`, price, `${def.magazine} rounds for the ${def.name}.`);
    }
    const price = WEAPON_PRICES[weaponId].price;
    if (!economy.trySpend(price, 'shop:weapon')) return this.denied(price);
    weapons.give(weaponId, def.magazine);
    return this.bought(weaponId, price, `${def.name} — comes with one clip.`);
  }

  buyFood(foodId: FoodId | null): PurchaseResult {
    const food = foodId ? FOOD[foodId] : null;
    if (!food) return { ok: false, message: 'Nothing to eat here.' };
    if (!this.game.systems.has('player')) return { ok: false, message: 'Nobody to feed.' };
    const economy = this.game.economy;
    if (!economy || !economy.trySpend(food.price, 'shop:food')) return this.denied(food.price);
    this.game.player.heal(food.heal);
    this.game.audio.play('food.eat');
    this.game.events.emit('shop:purchased', { itemId: food.id, price: food.price });
    this.game.events.emit('notification', { text: `${food.name}: +${food.heal} health`, kind: 'info' });
    return { ok: true, message: food.name };
  }

  private denied(price: number): PurchaseResult {
    this.game.audio.play('shop.denied');
    return { ok: false, message: `You need $${price.toLocaleString('en-US')}.` };
  }

  private bought(itemId: string, price: number, message: string): PurchaseResult {
    this.game.audio.play('shop.buy');
    this.game.events.emit('shop:purchased', { itemId, price });
    return { ok: true, message };
  }
}

function isBuyable(id: string): id is BuyableWeaponId {
  return (BUYABLE_WEAPONS as readonly string[]).includes(id);
}
