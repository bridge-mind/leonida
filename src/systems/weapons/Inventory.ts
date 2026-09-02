import type { WeaponId } from '../../core/peers';
import { WEAPON_ORDER, WEAPONS, isWeaponId } from '../../data/weapons';

export interface AmmoSlot {
  mag: number;
  reserve: number;
}

export interface InventorySave {
  current: WeaponId;
  slots: [WeaponId, number, number][];
}

/** Owned weapons with magazine + reserve ammo. Fists are always owned. */
export class Inventory {
  private readonly slots = new Map<WeaponId, AmmoSlot>();
  current: WeaponId = 'fists';

  constructor() {
    this.slots.set('fists', { mag: 0, reserve: 0 });
  }

  has(id: WeaponId): boolean {
    return this.slots.has(id);
  }

  slot(id: WeaponId): AmmoSlot | undefined {
    return this.slots.get(id);
  }

  get currentSlot(): AmmoSlot {
    return this.slots.get(this.current) ?? this.slots.get('fists')!;
  }

  /** Adds the weapon (filling the magazine first) or tops up ammo when already owned. Returns true when newly owned. */
  give(id: WeaponId, ammo: number): boolean {
    const def = WEAPONS[id];
    const existing = this.slots.get(id);
    if (existing) {
      this.addAmmo(id, ammo);
      return false;
    }
    const mag = Math.min(def.magazine, Math.max(0, Math.floor(ammo)));
    this.slots.set(id, { mag, reserve: Math.max(0, Math.floor(ammo) - mag) });
    return true;
  }

  addAmmo(id: WeaponId, amount: number): void {
    const s = this.slots.get(id);
    if (!s || id === 'fists') return;
    s.reserve = Math.max(0, s.reserve + Math.floor(amount));
  }

  /** Moves rounds from reserve into the magazine; returns the number loaded. */
  completeReload(id: WeaponId): number {
    const s = this.slots.get(id);
    if (!s) return 0;
    const need = WEAPONS[id].magazine - s.mag;
    const loaded = Math.min(need, s.reserve);
    s.mag += loaded;
    s.reserve -= loaded;
    return loaded;
  }

  canReload(id: WeaponId): boolean {
    const s = this.slots.get(id);
    return !!s && id !== 'fists' && s.mag < WEAPONS[id].magazine && s.reserve > 0;
  }

  /** Next / previous owned weapon in wheel order (wraps). */
  cycle(step: 1 | -1): WeaponId {
    const n = WEAPON_ORDER.length;
    let idx = WEAPON_ORDER.indexOf(this.current);
    for (let i = 0; i < n; i++) {
      idx = (idx + step + n) % n;
      const id = WEAPON_ORDER[idx];
      if (this.slots.has(id)) return id;
    }
    return this.current;
  }

  /** Weapon id -> reserve ammo, for save/stats. */
  summary(): Partial<Record<WeaponId, number>> {
    const out: Partial<Record<WeaponId, number>> = {};
    for (const [id, s] of this.slots) out[id] = s.reserve;
    return out;
  }

  serialize(): InventorySave {
    const slots: [WeaponId, number, number][] = [];
    for (const [id, s] of this.slots) slots.push([id, s.mag, s.reserve]);
    return { current: this.current, slots };
  }

  deserialize(data: unknown): void {
    const d = data as Partial<InventorySave> | null;
    if (!d || !Array.isArray(d.slots)) return;
    this.slots.clear();
    this.slots.set('fists', { mag: 0, reserve: 0 });
    for (const entry of d.slots) {
      if (!Array.isArray(entry) || typeof entry[0] !== 'string' || !isWeaponId(entry[0])) continue;
      const def = WEAPONS[entry[0]];
      const mag = Math.min(def.magazine, Math.max(0, Number(entry[1]) || 0));
      this.slots.set(entry[0], { mag, reserve: Math.max(0, Number(entry[2]) || 0) });
    }
    this.current = typeof d.current === 'string' && isWeaponId(d.current) && this.slots.has(d.current) ? d.current : 'fists';
  }
}
