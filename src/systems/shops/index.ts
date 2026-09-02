import { Vector3 } from 'three';
import type { IPed } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { IRobberyPeer, IShopsPeer } from '../../core/peers';
import { Random } from '../../core/Random';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { FOOD, type BuyableWeaponId } from '../../data/prices';
import { STORES, type StoreDef } from '../../data/shops';
import { Phone } from '../../ui/phone/Phone';
import { GunStorePanel, type CatalogueRow, type PurchaseResult } from '../../ui/shops/GunStore';
import { createStoreInterior } from '../../world/interiors/stores';
import { ShopsConfig } from './config';
import './events';
import { PromptSlot } from './prompt';
import { Purchases } from './Purchases';
import { Respray } from './Respray';
import { StoreRuntime } from './StoreRuntime';

export type ShopItem = BuyableWeaponId | 'armor';

/**
 * Stores: registers every store interior, staffs it while the player is inside,
 * runs the counter interactions (gun store UI, food, safehouse save), the
 * Vice Sprays drive-in, and hosts the phone. Robbery and the chop shop are
 * separate systems that read store state through `runtime()` / `activeStore()`.
 */
export class ShopSystem implements GameSystem, IShopsPeer {
  private readonly stores = new Map<string, StoreRuntime>();
  private current: StoreRuntime | null = null;
  private readonly prompt: PromptSlot;
  private readonly gunStore: GunStorePanel;
  private readonly purchases: Purchases;
  readonly phone: Phone;
  private readonly respray: Respray;
  private readonly rng = Random.unseeded();
  private readonly origin = new Vector3();
  private readonly tmp = new Vector3();
  private readonly unsubs: Array<() => void> = [];
  /** Closed-until days from a save that arrived before the interiors were registered. */
  private savedClosed: Record<string, number> = {};

  constructor(private readonly game: Game) {
    this.prompt = new PromptSlot(game.state);
    this.purchases = new Purchases(game);
    this.gunStore = new GunStorePanel(game, { rows: () => this.rows(), purchase: (item) => this.purchase(item) });
    this.phone = new Phone(game);
    this.respray = new Respray(game, this.rng);
  }

  init(): void {
    // The interiors system boots after us (PRIORITY.interiors > shops), so registration waits for game:ready.
    this.registerInteriors();
    const ev = this.game.events;
    this.unsubs.push(
      ev.on('interior:entered', ({ id }) => this.onEntered(id)),
      ev.on('interior:exited', ({ id }) => this.onExited(id)),
      ev.on('time:dayChanged', () => this.refreshBlips()),
      ev.on('game:ready', () => {
        this.registerInteriors();
        this.refreshBlips();
      }),
    );
    this.game.debug.register('phone', {
      open: (): boolean => {
        this.phone.open();
        return this.phone.isOpen;
      },
      close: (): void => this.phone.close(),
      snap: (): Promise<string> => this.phone.snap(),
      app: (id: unknown): void => this.phone.openApp(String(id)),
      feed: (): string[] => this.phone.feed.posts.map((p) => `${p.handle}: ${p.text}`),
    });
    this.game.debug.register('respray', {
      spray: (): Promise<boolean> => this.respray.spray(),
      position: (): { x: number; y: number; z: number } => ({
        x: this.respray.position.x,
        y: this.respray.position.y,
        z: this.respray.position.z,
      }),
    });
  }

  private registerInteriors(): void {
    const interiors = this.game.interiors;
    if (!interiors || this.stores.size > 0) return;
    this.origin.copy(interiors.framework.origin);
    // Slot 0 is the casino; each store gets its own strip of the interior origin.
    let slot = 1;
    for (const def of STORES) {
      if (!def.hasInterior) continue;
      const interior = createStoreInterior(def, slot++);
      if (!interior) continue;
      interiors.register(interior.definition);
      this.stores.set(def.id, new StoreRuntime(def, interior));
    }
    this.applySavedClosed();
  }

  // --- IShopsPeer ----------------------------------------------------------------

  openGunStore(): void {
    if (this.gunStore.isOpen) return;
    this.gunStore.open();
  }

  isStoreClosed(storeId: string): boolean {
    return this.stores.get(storeId)?.isClosed(this.game.time.day) ?? false;
  }

  // --- API for robbery / chop shop -----------------------------------------------

  /** Store the player is currently inside, if it is one of ours. */
  activeStore(): StoreRuntime | null {
    return this.current;
  }

  runtime(storeId: string): StoreRuntime | undefined {
    return this.stores.get(storeId);
  }

  get definitions(): readonly StoreDef[] {
    return STORES;
  }

  /** World position of a layout point of a store. */
  worldPoint(local: { x: number; y: number; z: number }, out: Vector3): Vector3 {
    return out.set(this.origin.x + local.x, this.origin.y + local.y, this.origin.z + local.z);
  }

  /** Puts the clerk's hands up; returns the clerk so the robbery can track him. */
  beginRobbery(storeId: string): IPed | null {
    const rt = this.stores.get(storeId);
    if (!rt?.clerk || !rt.clerkBrain) return null;
    rt.clerkBrain.beginRobbery(rt.clerk);
    return rt.clerk;
  }

  closeStore(storeId: string): void {
    const rt = this.stores.get(storeId);
    if (!rt) return;
    rt.closedUntilDay = this.game.time.day + ShopsConfig.closedDays;
    this.game.events.emit('shops:closed', { storeId, untilDay: rt.closedUntilDay });
    this.refreshBlips();
  }

  // --- Purchases -----------------------------------------------------------------

  rows(): CatalogueRow[] {
    return this.purchases.rows();
  }

  purchase(item: string): PurchaseResult {
    return this.purchases.buy(item);
  }

  /** Buys the current store's food item and heals. */
  buyFood(): PurchaseResult {
    return this.purchases.buyFood(this.current?.def.food ?? null);
  }

  // --- Loop ------------------------------------------------------------------------

  update(dt: number): void {
    this.phone.update();
    this.gunStore.update();
    this.respray.update();
    const rt = this.current;
    if (!rt) return;
    const weapons = this.game.weapons;
    const armed = weapons !== undefined && weapons.current !== 'fists';
    rt.update(dt, armed);
    this.updateCounter(rt);
  }

  private updateCounter(rt: StoreRuntime): void {
    if (!this.game.systems.has('player') || this.gunStore.isOpen || this.game.input.isBlocked) {
      this.prompt.set(null);
      return;
    }
    const robbery = this.game.systems.get<GameSystem & IRobberyPeer>('robbery');
    if (robbery?.activeStore) {
      this.prompt.set(null);
      return;
    }
    const player = this.game.player;
    const layout = rt.layout;
    const target = rt.def.kind === 'safehouse' ? layout.bed : layout.counter;
    if (!target) {
      this.prompt.set(null);
      return;
    }
    this.worldPoint(target, this.tmp);
    const dx = this.tmp.x - player.position.x;
    const dz = this.tmp.z - player.position.z;
    const radius = Math.max(layout.interactRadius, ShopsConfig.counterRadius);
    if (dx * dx + dz * dz > radius * radius) {
      this.prompt.set(null);
      return;
    }
    let text: string | null = null;
    switch (rt.def.kind) {
      case 'gun':
        text = 'Browse weapons';
        break;
      case 'convenience':
      case 'gas':
      case 'food': {
        const food = rt.def.food ? FOOD[rt.def.food] : null;
        text = food ? `Buy ${food.name} $${food.price}` : null;
        break;
      }
      case 'safehouse':
        text = 'Save game';
        break;
      default:
        text = null;
    }
    this.prompt.set(text);
    if (text === null || !this.game.input.wasPressed('KeyE')) return;
    if (rt.def.kind === 'gun') this.openGunStore();
    else if (rt.def.kind === 'safehouse') {
      if (this.game.save.save()) this.game.audio.play('save.complete');
    } else this.buyFood();
  }

  // --- Interior hooks ----------------------------------------------------------------

  private onEntered(id: string): void {
    const rt = this.stores.get(id);
    if (!rt) return;
    this.current = rt;
    const name = rt.def.name;
    rt.staff(this.game, this.origin, this.rng, {
      onClerkShotgun: () => {
        this.game.audio.play('weapon.shotgun.reload');
        this.game.events.emit('notification', { text: 'Clerk: "Not in my store!"', kind: 'warning' });
        this.game.events.emit('shops:hostile', { storeId: id, role: 'clerk' });
      },
      onGuardWarning: () => this.game.events.emit('notification', { text: 'Guard: "Drop it. Now."', kind: 'warning' }),
      onGuardHostile: () => this.game.events.emit('shops:hostile', { storeId: id, role: 'guard' }),
    });
    this.game.audio.play('store.bell');
    if (rt.def.kind === 'gun') this.game.events.emit('notification', { text: `${name}: talk to the clerk to browse.`, kind: 'info' });
  }

  private onExited(id: string): void {
    const rt = this.stores.get(id);
    if (rt) rt.unstaff(this.game);
    if (this.current === rt) this.current = null;
    this.prompt.set(null);
    this.gunStore.close();
  }

  /** Landmark blips already exist (navigation adds one per landmark); we only relabel ours. */
  private refreshBlips(): void {
    const nav = this.game.navigation;
    if (!nav || !this.game.systems.has('world')) return;
    const day = this.game.time.day;
    for (const def of STORES) {
      // The chop shop labels its own blip with today's wanted archetype; it subscribes to the same
      // events before us, so writing it here would clobber that label every time.
      if (def.kind === 'chopshop') continue;
      const landmark = this.game.world.landmarks.byId(def.landmarkId);
      if (!landmark) continue;
      const rt = this.stores.get(def.id);
      const closed = rt?.isClosed(day) ?? false;
      nav.addBlip({ id: landmark.id, kind: landmark.kind, position: landmark.door, label: closed ? `${def.name} (closed)` : def.name });
    }
  }

  dispose(): void {
    for (const u of this.unsubs) u();
    this.unsubs.length = 0;
    for (const rt of this.stores.values()) rt.unstaff(this.game);
    this.current = null;
    this.prompt.set(null);
    this.gunStore.dispose();
    this.phone.dispose();
    this.respray.dispose();
  }

  serialize(): unknown {
    const closed: Record<string, number> = {};
    for (const [id, rt] of this.stores) if (rt.closedUntilDay > 0) closed[id] = rt.closedUntilDay;
    return { closed };
  }

  deserialize(data: unknown): void {
    const d = data as { closed?: Record<string, number> } | undefined;
    this.savedClosed = d?.closed ?? {};
    this.applySavedClosed();
  }

  private applySavedClosed(): void {
    for (const rt of this.stores.values()) {
      const day = this.savedClosed[rt.def.id];
      rt.closedUntilDay = typeof day === 'number' ? day : 0;
    }
  }

  debugActions = {
    open: (which: unknown = 'gunstore'): boolean => {
      if (String(which) === 'gunstore') this.openGunStore();
      else this.phone.open();
      return this.gunStore.isOpen || this.phone.isOpen;
    },
    closeUi: (): void => this.gunStore.close(),
    buy: (item: unknown): PurchaseResult => this.purchase(String(item)),
    food: (): PurchaseResult => this.buyFood(),
    close: (storeId: unknown): boolean => {
      this.closeStore(String(storeId));
      return this.isStoreClosed(String(storeId));
    },
    reopen: (storeId: unknown): void => {
      const rt = this.stores.get(String(storeId));
      if (rt) rt.closedUntilDay = 0;
      this.refreshBlips();
    },
    list: (): string[] => [...this.stores.keys()],
    status: (): Record<string, unknown> => ({
      current: this.current?.def.id ?? null,
      clerk: this.current?.clerk?.id ?? null,
      clerkMode: this.current?.clerkBrain?.mode ?? null,
      shotgun: this.current?.shotgunRolled ?? false,
      guards: this.current?.guards.length ?? 0,
      closed: [...this.stores.values()].filter((rt) => rt.isClosed(this.game.time.day)).map((rt) => rt.def.id),
      gunStoreOpen: this.gunStore.isOpen,
    }),
    /** Forces the clerk's shotgun roll for the current visit (autotests). */
    armClerk: (armed: unknown = true): void => {
      const rt = this.current;
      if (!rt) return;
      rt.shotgunOverride = Boolean(armed);
      this.onEntered(rt.def.id);
    },
  };
}

const module: SystemModule = {
  id: 'shops',
  priority: PRIORITY.shops,
  create: (game) => new ShopSystem(game),
};

export default module;
