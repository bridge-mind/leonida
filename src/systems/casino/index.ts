import { Vector3 } from 'three';
import { Blackjack } from '../../casino/Blackjack';
import { LuckyWheel, rollMystery, type WheelSegment } from '../../casino/LuckyWheel';
import { Config } from '../../core/Config';
import type { V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { ICasinoPeer } from '../../core/peers';
import { Random } from '../../core/Random';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { BlackjackPanel } from '../../ui/casino/BlackjackPanel';
import { CashierPanel } from '../../ui/casino/CashierPanel';
import type { CasinoPanel } from '../../ui/casino/Panel';
import { RoulettePanel } from '../../ui/casino/RoulettePanel';
import { SlotsPanel } from '../../ui/casino/SlotsPanel';
import { WheelPanel } from '../../ui/casino/WheelPanel';
import { createCasinoInterior, type CasinoStage, type StationId } from '../../world/interiors/CasinoInterior';
import type { InteriorDefinition } from '../../world/interiors/Interiors';
import { CasinoConfig } from './config';
import './events';
import { CasinoGuards } from './guards';
import { CasinoRig } from './rig';

export type CasinoGame = 'blackjack' | 'roulette' | 'slots' | 'wheel';

/**
 * Leonida Palms (SYSTEM_PROMPT §11): chips, the interior, the table UIs, guards and
 * the daily Lucky Wheel. Pure game rules live in `src/casino/`.
 */
export class CasinoSystem implements GameSystem, ICasinoPeer {
  private balance = 0;
  private lockedUntilDay = 0;
  private inside = false;
  private kickPending = false;
  private crimePending = false;
  private warnedMissing = false;
  private activePanel: CasinoPanel | null = null;
  private readonly rngSource = Random.unseeded();
  readonly rng = (): number => this.rngSource.next();
  readonly rig = new CasinoRig();
  readonly tables = {
    low: new Blackjack(CasinoConfig.tables.low, this.rng),
    high: new Blackjack(CasinoConfig.tables.high, this.rng),
  };
  readonly wheel = new LuckyWheel(this.rng);
  readonly stage: CasinoStage;
  private readonly interiorDefinition: InteriorDefinition;
  private readonly guards: CasinoGuards;
  private readonly origin: V3 = { x: Config.interiors.origin.x, y: Config.interiors.origin.y, z: Config.interiors.origin.z };
  private readonly tmp = new Vector3();

  constructor(readonly game: Game) {
    this.guards = new CasinoGuards(game, this.origin);
    const interior = createCasinoInterior({
      game,
      onStation: (station) => this.openStation(station),
      isUiOpen: () => this.isUiOpen(),
    });
    this.stage = interior.stage;
    this.interiorDefinition = interior.definition;
  }

  init(): void {
    const g = this.game;
    if (!g.economy && !this.warnedMissing) {
      this.warnedMissing = true;
      console.warn('[casino] economy system missing: chips cannot be bought or cashed out');
    }
    // Interiors (priority 540) initialise after us; register now if present, else once everything is ready.
    if (g.interiors) this.registerInterior();
    else g.events.once('game:ready', () => this.registerInterior());
    g.events.on('interior:entered', ({ id }) => this.onInteriorEntered(id));
    g.events.on('interior:exited', ({ id }) => this.onInteriorExited(id));
    g.events.on('weapon:fired', ({ shooter }) => {
      if (shooter === 'player' && g.state.interior === CasinoConfig.interiorId) this.onPlayerFired();
    });
    g.events.on('time:dayChanged', ({ day }) => {
      if (day >= this.lockedUntilDay) this.guards.reset();
    });
    this.publish(0);
  }

  private registerInterior(): void {
    const interiors = this.game.interiors;
    if (!interiors) {
      console.warn('[casino] interiors system missing: Leonida Palms has no door');
      return;
    }
    const origin = interiors.framework.origin;
    this.origin.x = origin.x;
    this.origin.y = origin.y;
    this.origin.z = origin.z;
    interiors.register(this.interiorDefinition);
  }

  // --- ICasinoPeer ---------------------------------------------------------------

  get chips(): number {
    return this.balance;
  }

  buyChips(amount: number): boolean {
    const n = Math.round(amount);
    const economy = this.game.economy;
    if (n <= 0 || !economy) return false;
    if (!economy.trySpend(n, 'casino chips')) return false;
    this.addChips(n);
    this.game.audio.play('casino.cashier');
    return true;
  }

  cashOut(amount: number): boolean {
    const n = Math.round(amount);
    const economy = this.game.economy;
    if (n <= 0 || n > this.balance || !economy) return false;
    this.addChips(-n);
    economy.add(n, 'casino cashout');
    this.game.audio.play('casino.cashier');
    return true;
  }

  play(game: CasinoGame): void {
    if (this.game.state.interior !== CasinoConfig.interiorId) {
      console.warn('[casino] play() ignored: player is not inside Leonida Palms');
      return;
    }
    const station: StationId = game === 'blackjack' ? 'blackjackLow' : game;
    this.openStation(station);
  }

  // --- Public helpers (panels, debug, missions) ----------------------------------

  get isLocked(): boolean {
    return this.game.time.day < this.lockedUntilDay;
  }

  get isInside(): boolean {
    return this.inside;
  }

  get panel(): CasinoPanel | null {
    return this.activePanel;
  }

  /** Teleports the player to the casino door and enters the interior. */
  async enter(): Promise<boolean> {
    const interiors = this.game.interiors;
    if (!interiors) return false;
    const door = this.doorPosition(0);
    if (door && this.game.systems.has('player')) this.game.player.teleport(door, this.doorHeading() + Math.PI);
    return interiors.enter(CasinoConfig.interiorId);
  }

  /** Deducts chips for a bet; false (nothing changes) when the balance cannot cover it. */
  takeChips(amount: number): boolean {
    if (!Number.isFinite(amount) || amount <= 0 || amount > this.balance) return false;
    this.addChips(-amount);
    return true;
  }

  giveChips(amount: number): void {
    if (amount > 0) this.addChips(amount);
  }

  private addChips(delta: number): void {
    const rounded = Math.round(delta);
    if (rounded === 0) return;
    this.balance = Math.max(0, this.balance + rounded);
    this.publish(rounded);
  }

  private publish(delta: number): void {
    this.game.state.chips = this.balance;
    this.game.events.emit('casino:chipsChanged', { chips: this.balance, delta });
  }

  /** Emits the cross-system win/lose events and notifies on big wins. */
  settleRound(game: CasinoGame, net: number): void {
    const stats = this.game.state.stats;
    if (net > 0) {
      stats.casinoWon = (stats.casinoWon ?? 0) + net;
      this.game.events.emit('casino:won', { game, amount: net });
      if (net >= CasinoConfig.bigWin) {
        this.game.audio.play('casino.win.big');
        this.notify(`${CasinoConfig.name}: you won ${net.toLocaleString('en-US')} chips at ${game}!`, 'money');
      }
    } else if (net < 0) {
      stats.casinoLost = (stats.casinoLost ?? 0) - net;
      this.game.events.emit('casino:lost', { game, amount: -net });
    }
  }

  /** Applies a Lucky Wheel segment. Returns the message to show the player. */
  applyWheelPrize(segment: WheelSegment): string {
    const g = this.game;
    const prize = segment.prize;
    switch (prize.kind) {
      case 'cash':
        g.economy?.add(prize.amount, 'lucky wheel');
        this.settleRound('wheel', prize.amount);
        return `You won $${prize.amount.toLocaleString('en-US')}!`;
      case 'chips':
        this.giveChips(prize.amount);
        this.settleRound('wheel', prize.amount);
        return `You won ${prize.amount.toLocaleString('en-US')} chips!`;
      case 'mystery': {
        const m = rollMystery(this.rng);
        const weapons = g.weapons;
        if (m.kind === 'weapon' && weapons) {
          weapons.give(m.weaponId, m.ammo);
          return `Mystery prize: an SMG with ${m.ammo} rounds.`;
        }
        if (m.kind === 'ammo' && weapons) {
          weapons.addAmmo(m.weaponId, m.ammo);
          return `Mystery prize: ${m.ammo} pistol rounds.`;
        }
        if (g.systems.has('player')) g.player.addArmor(100);
        return 'Mystery prize: full body armor.';
      }
      case 'vehicle': {
        const vehicles = g.vehicles;
        const spot = this.doorPosition(CasinoConfig.wheel.vehicleOffset);
        if (vehicles && spot) {
          vehicles.spawn('sports', spot, this.doorHeading(), { parked: true });
          this.notify('A sports car is waiting outside.', 'info');
          return 'You won a sports car! It is parked outside.';
        }
        g.economy?.add(CasinoConfig.wheel.vehicleFallbackCash, 'lucky wheel');
        this.settleRound('wheel', CasinoConfig.wheel.vehicleFallbackCash);
        return `You won the car! Paid out as $${CasinoConfig.wheel.vehicleFallbackCash.toLocaleString('en-US')}.`;
      }
    }
  }

  notify(text: string, kind: 'info' | 'money' | 'warning' = 'info'): void {
    const ui = this.game.ui;
    if (ui) ui.notify(text, kind);
    else this.game.events.emit('notification', { text, kind });
  }

  // --- Interior / guards ---------------------------------------------------------

  private isUiOpen(): boolean {
    return this.activePanel !== null || this.game.input.isBlocked || this.game.ui?.isModal === true;
  }

  private openStation(station: StationId): void {
    if (!this.inside || this.isUiOpen()) return;
    let panel: CasinoPanel;
    switch (station) {
      case 'cashier':
        panel = new CashierPanel(this);
        break;
      case 'blackjackLow':
        panel = new BlackjackPanel(this, this.tables.low, 'low');
        break;
      case 'blackjackHigh':
        panel = new BlackjackPanel(this, this.tables.high, 'high');
        break;
      case 'roulette':
        panel = new RoulettePanel(this);
        break;
      case 'slots':
        panel = new SlotsPanel(this);
        break;
      case 'wheel':
        panel = new WheelPanel(this);
        break;
    }
    this.activePanel = panel;
    panel.open();
  }

  /** Called by panels when they close themselves. */
  onPanelClosed(panel: CasinoPanel): void {
    if (this.activePanel === panel) this.activePanel = null;
  }

  closePanel(): void {
    this.activePanel?.close(true);
    this.activePanel = null;
  }

  private onInteriorEntered(id: string): void {
    if (id !== CasinoConfig.interiorId) return;
    if (this.isLocked) {
      this.kickPending = true;
      this.notify(`${CasinoConfig.name}: you are banned until tomorrow.`, 'warning');
      this.game.audio.play('door.locked');
      return;
    }
    this.inside = true;
    this.guards.spawn();
    // Ambient bed: the audio system's AmbientManager already plays `casino.ambient.loop` from `state.interior`.
    this.game.events.emit('casino:entered', { id: 'casino' });
  }

  private onInteriorExited(id: string): void {
    if (id !== CasinoConfig.interiorId) return;
    const wasInside = this.inside;
    this.inside = false;
    this.closePanel();
    this.guards.despawn();
    if (wasInside) this.game.events.emit('casino:exited', { id: 'casino' });
    if (this.crimePending) {
      this.crimePending = false;
      const door = this.doorPosition(0) ?? { x: 0, y: 0, z: 0 };
      this.game.events.emit('crime:committed', { type: 'rob_store', position: door, actor: 'player', loud: false });
    }
  }

  private onPlayerFired(): void {
    if (this.guards.hostile) return;
    this.guards.alert();
    this.lockedUntilDay = this.game.time.day + CasinoConfig.banDays;
    this.crimePending = true;
    this.closePanel();
    this.game.audio.play('casino.guard.alert');
    this.game.events.emit('casino:locked', { untilDay: this.lockedUntilDay });
    this.notify(`${CasinoConfig.name} security: weapons are not tolerated. You are banned for a day.`, 'warning');
  }

  private doorPosition(offset: number): V3 | null {
    if (!this.game.systems.has('world')) return null;
    const landmark = this.game.world.landmarks.byId(CasinoConfig.landmarkId);
    if (!landmark) return null;
    const h = landmark.doorHeading;
    this.tmp.set(landmark.door.x + Math.sin(h) * offset, landmark.door.y, landmark.door.z + Math.cos(h) * offset);
    return { x: this.tmp.x, y: this.tmp.y, z: this.tmp.z };
  }

  private doorHeading(): number {
    if (!this.game.systems.has('world')) return 0;
    return this.game.world.landmarks.byId(CasinoConfig.landmarkId)?.doorHeading ?? 0;
  }

  // --- Loop / lifecycle ------------------------------------------------------------

  update(): void {
    if (this.kickPending) {
      const interiors = this.game.interiors;
      if (!interiors) {
        this.kickPending = false;
      } else if (!interiors.framework.transitioning) {
        this.kickPending = false;
        void interiors.exit();
      }
    }
  }

  dispose(): void {
    this.closePanel();
    this.guards.despawn();
  }

  serialize(): unknown {
    return { chips: this.balance, lastWheelDay: this.wheel.lastSpinDay, lockedUntilDay: this.lockedUntilDay };
  }

  deserialize(data: unknown): void {
    const d = data as { chips?: number; lastWheelDay?: number; lockedUntilDay?: number } | null;
    if (!d) return;
    if (typeof d.chips === 'number') this.balance = Math.max(0, Math.round(d.chips));
    if (typeof d.lastWheelDay === 'number') this.wheel.lastSpinDay = d.lastWheelDay;
    if (typeof d.lockedUntilDay === 'number') this.lockedUntilDay = d.lockedUntilDay;
    this.publish(0);
  }

  debugActions = {
    enter: (): Promise<boolean> => this.enter(),
    buy: (amount: unknown): number => {
      this.buyChips(Number(amount));
      return this.balance;
    },
    cashout: (amount: unknown): number => {
      this.cashOut(Number(amount));
      return this.balance;
    },
    chips: (): number => this.balance,
    play: (game: unknown): void => this.play(String(game) as CasinoGame),
    rig: (game: unknown, value: unknown): string => {
      this.rig.set(String(game), value);
      return `rigged ${String(game)}`;
    },
    close: (): void => this.closePanel(),
    lock: (days?: unknown): number => {
      this.lockedUntilDay = this.game.time.day + (days === undefined ? CasinoConfig.banDays : Number(days));
      return this.lockedUntilDay;
    },
    unlock: (): void => {
      this.lockedUntilDay = 0;
      this.guards.reset();
    },
  };
}

const module: SystemModule = {
  id: 'casino',
  priority: PRIORITY.casino,
  create: (game) => new CasinoSystem(game),
};

export default module;
