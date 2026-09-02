import { Vector3 } from 'three';
import type { Entity, IPed } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { IRobberyPeer } from '../../core/peers';
import { Random } from '../../core/Random';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import type { ShopSystem } from '../shops';
import type { StoreRuntime } from '../shops/StoreRuntime';
import { RobberyConfig as C } from './config';
import { RobberyClock } from './progress';

interface ActiveRobbery {
  store: StoreRuntime;
  clerk: IPed;
  clock: RobberyClock;
  completed: boolean;
  /** Any shot fired inside makes the eventual crime report "loud". */
  loud: boolean;
  lineTimer: number;
}

/**
 * Store hold-ups. Aim a drawn gun at the clerk for a second: he raises his hands
 * and starts filling a bag over `bagSeconds` (gunshots hurry him). The cash lands
 * on the register as a pickup; the store closes for the day; walking out reports
 * the crime so the wanted system can respond (this system never touches wanted).
 */
export class RobberySystem implements GameSystem, IRobberyPeer {
  private active: ActiveRobbery | null = null;
  private aimTime = 0;
  private sinceAim = Number.POSITIVE_INFINITY;
  private aimingAt: Entity | null = null;
  private readonly rng = Random.unseeded();
  private readonly tmp = new Vector3();
  private readonly unsubs: Array<() => void> = [];
  /** Store id whose robbery still needs reporting when the player walks out. */
  private pendingCrime: StoreRuntime | null = null;

  constructor(private readonly game: Game) {}

  init(): void {
    const ev = this.game.events;
    this.unsubs.push(
      ev.on('weapon:aimedAt', ({ target }) => this.onAimedAt(target)),
      ev.on('weapon:fired', ({ shooter }) => {
        if (shooter === 'player') this.onPlayerShot();
      }),
      ev.on('interior:exited', ({ id }) => this.onExited(id)),
      ev.on('player:died', () => this.abort()),
    );
  }

  dispose(): void {
    for (const u of this.unsubs) u();
    this.unsubs.length = 0;
    this.abort();
  }

  // --- IRobberyPeer ---------------------------------------------------------------

  get activeStore(): string | null {
    return this.active?.store.def.id ?? null;
  }

  // --- Flow -----------------------------------------------------------------------

  private get shops(): ShopSystem | undefined {
    return this.game.systems.get<ShopSystem>('shops');
  }

  private onAimedAt(target: Entity): void {
    const store = this.shops?.activeStore();
    if (!store || !store.clerk || target !== store.clerk) return;
    const weapons = this.game.weapons;
    if (!weapons || weapons.current === 'fists') return;
    this.aimingAt = target;
    this.sinceAim = 0;
  }

  update(dt: number): void {
    const store = this.shops?.activeStore() ?? null;
    if (!this.active) {
      this.trackAim(dt, store);
      return;
    }
    const a = this.active;
    if (a.completed) return;
    if (a.clock.tick(dt)) {
      this.complete(a);
      return;
    }
    this.game.ui?.progressBar.show('Filling bag', a.clock.fraction);
    a.lineTimer -= dt;
    if (a.lineTimer <= 0) {
      a.lineTimer = 4 + this.rng.range(0, 3);
      this.game.events.emit('notification', { text: `Clerk: ${this.rng.pick(C.clerkLines)}`, kind: 'info' });
    }
  }

  private trackAim(dt: number, store: StoreRuntime | null): void {
    this.sinceAim += dt;
    if (!store || !store.clerk || !store.def.registerCash || this.aimingAt !== store.clerk || this.sinceAim > C.aimGapSeconds) {
      this.aimTime = 0;
      if (this.sinceAim > C.aimGapSeconds) this.aimingAt = null;
      return;
    }
    this.aimTime += dt;
    if (this.aimTime >= C.aimSeconds) this.start(store.def.id);
  }

  /** Starts a hold-up in `storeId` (player must be inside with the clerk present). */
  start(storeId: string): boolean {
    if (this.active) return false;
    const shops = this.shops;
    const store = shops?.runtime(storeId);
    if (!shops || !store || !store.def.registerCash || shops.activeStore() !== store) return false;
    const clerk = shops.beginRobbery(storeId);
    if (!clerk) return false;
    this.active = {
      store,
      clerk,
      clock: new RobberyClock(C.bagSeconds, C.shotCut, C.maxShots),
      completed: false,
      loud: false,
      lineTimer: 1.5,
    };
    this.pendingCrime = store;
    this.aimTime = 0;
    this.game.audio.play('register.open');
    this.game.audio.play('robbery.bagfill.loop');
    this.game.ui?.progressBar.show('Filling bag', 0);
    this.game.events.emit('robbery:started', { storeId });
    return true;
  }

  private onPlayerShot(): void {
    const a = this.active;
    if (!a) return;
    a.loud = true;
    if (!a.completed && a.clock.onShot()) {
      this.game.events.emit('notification', { text: 'Clerk: "Okay! Okay! Faster!"', kind: 'warning' });
    }
  }

  private complete(a: ActiveRobbery): void {
    a.completed = true;
    const range = a.store.def.registerCash ?? { min: 200, max: 800 };
    const amount = Math.round(this.rng.int(range.min, range.max) / 10) * 10;
    const shops = this.shops;
    if (shops) {
      shops.worldPoint(a.store.layout.register, this.tmp);
      // Cash pickups snap to the ground under them; the register sits on the counter, so drop it at counter height.
      this.game.pickups?.spawn('cash', this.tmp, amount);
      shops.closeStore(a.store.def.id);
    }
    this.game.ui?.progressBar.hide();
    this.game.audio.play('register.cash');
    this.game.audio.play('robbery.complete');
    this.game.events.emit('notification', { text: `Bag's full: $${amount.toLocaleString('en-US')}. Grab it and go.`, kind: 'money' });
    this.game.events.emit('robbery:completed', { storeId: a.store.def.id, amount });
  }

  private onExited(id: string): void {
    const pending = this.pendingCrime;
    const a = this.active;
    this.game.ui?.progressBar.hide();
    this.active = null;
    this.aimTime = 0;
    this.aimingAt = null;
    if (!pending || pending.def.id !== id) {
      this.pendingCrime = null;
      return;
    }
    this.pendingCrime = null;
    const p = this.game.systems.has('player') ? this.game.player.position : this.tmp.set(0, 0, 0);
    this.game.events.emit('crime:committed', {
      type: 'rob_store',
      position: { x: p.x, y: p.y, z: p.z },
      actor: 'player',
      loud: a?.loud ?? false,
    });
  }

  private abort(): void {
    if (!this.active) return;
    this.game.ui?.progressBar.hide();
    this.active = null;
    this.pendingCrime = null;
    this.aimTime = 0;
  }

  debugActions = {
    start: (storeId: unknown): boolean => this.start(String(storeId ?? this.shops?.activeStore()?.def.id ?? '')),
    status: (): Record<string, unknown> => ({
      active: this.activeStore,
      fraction: this.active?.clock.fraction ?? 0,
      secondsLeft: this.active?.clock.secondsLeft ?? 0,
      shots: this.active?.clock.shots ?? 0,
      completed: this.active?.completed ?? false,
      loud: this.active?.loud ?? false,
      aimTime: this.aimTime,
      pendingCrime: this.pendingCrime?.def.id ?? null,
    }),
    /** Skips ahead `seconds` of bag time (autotests). */
    skip: (seconds: unknown = 12): boolean => {
      const a = this.active;
      if (!a || a.completed) return false;
      if (a.clock.tick(Number(seconds))) this.complete(a);
      return a.completed;
    },
    abort: (): void => this.abort(),
  };
}

const module: SystemModule = {
  id: 'robbery',
  priority: PRIORITY.robbery,
  create: (game) => new RobberySystem(game),
};

export default module;
