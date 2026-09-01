import { Config } from '../../core/Config';
import type { Game } from '../../core/Game';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';

/**
 * Minimal wallet. Agent 9 takes ownership in Stage 3 and extends it (tick
 * animation, stats, fees) without changing `cash`, `add`, `trySpend`.
 */
export class EconomySystem implements GameSystem {
  private balance = Config.economy.startCash;
  /** Lifetime totals for the stats screen. */
  earned = 0;
  spent = 0;

  constructor(private readonly game: Game) {}

  init(): void {
    this.publish(0);
  }

  get cash(): number {
    return this.balance;
  }

  /** Adds (or with a negative amount, removes without a limit check) cash. */
  add(amount: number, reason: string): void {
    if (!Number.isFinite(amount) || amount === 0) return;
    const rounded = Math.round(amount);
    this.balance = Math.max(0, this.balance + rounded);
    if (rounded > 0) this.earned += rounded;
    else this.spent += -rounded;
    this.publish(rounded);
    this.game.events.emit('economy:cashChanged', { cash: this.balance, delta: rounded, reason });
    this.game.audio.play(rounded > 0 ? 'cash.gain' : 'cash.loss');
  }

  /** Returns false (and changes nothing) when the wallet cannot cover `amount`. */
  trySpend(amount: number, reason: string): boolean {
    const rounded = Math.round(amount);
    if (rounded < 0) return false;
    if (rounded > this.balance) {
      this.game.audio.play('shop.denied');
      return false;
    }
    this.add(-rounded, reason);
    return true;
  }

  canAfford(amount: number): boolean {
    return Math.round(amount) <= this.balance;
  }

  private publish(delta: number): void {
    this.game.state.cash = this.balance;
    this.game.state.cashDelta = delta;
    this.game.state.stats.cashEarned = this.earned;
    this.game.state.stats.cashSpent = this.spent;
  }

  serialize(): unknown {
    return { cash: this.balance, earned: this.earned, spent: this.spent };
  }

  deserialize(data: unknown): void {
    const d = data as { cash?: number; earned?: number; spent?: number };
    if (typeof d?.cash === 'number') this.balance = d.cash;
    if (typeof d?.earned === 'number') this.earned = d.earned;
    if (typeof d?.spent === 'number') this.spent = d.spent;
    this.publish(0);
  }

  debugActions = {
    add: (amount: unknown): number => {
      this.add(Number(amount), 'debug');
      return this.balance;
    },
    set: (amount: unknown): number => {
      this.balance = Math.max(0, Math.round(Number(amount)));
      this.publish(0);
      return this.balance;
    },
    get: (): number => this.balance,
  };
}

const module: SystemModule = {
  id: 'economy',
  priority: PRIORITY.economy,
  create: (game) => new EconomySystem(game),
};

export default module;
