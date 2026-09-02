import { Config } from '../../core/Config';
import type { Game } from '../../core/Game';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { EconomyConfig } from './config';

interface EconomySave {
  cash?: number;
  earned?: number;
  spent?: number;
  fromRobberies?: number;
  fromChopShop?: number;
}

/**
 * The wallet. `cash`, `add`, `trySpend`, `canAfford` are the public contract every
 * other system uses; on top of that it tracks per-source income for the stats
 * screen and charges the hospital bill when the player is wasted.
 */
export class EconomySystem implements GameSystem {
  private balance = Config.economy.startCash;
  /** Lifetime totals for the stats screen. */
  earned = 0;
  spent = 0;
  fromRobberies = 0;
  fromChopShop = 0;
  private readonly unsubs: Array<() => void> = [];

  constructor(private readonly game: Game) {}

  init(): void {
    this.publish(0);
    this.unsubs.push(
      // Single owner of the 'wasted' hospital bill; the player system only bills 'bail' on 'busted'.
      this.game.events.on('player:respawned', (e) => {
        if (e.reason !== 'wasted') return;
        this.fee(EconomyConfig.hospitalFraction, 'hospital', EconomyConfig.hospitalMin);
      }),
      // Income by source is counted from the source events, not from `add` reasons:
      // robbery cash arrives through a generic pickup, so the reason there is just 'pickup'.
      this.game.events.on('robbery:completed', ({ amount }) => {
        this.fromRobberies += amount;
        this.publish(0);
      }),
      this.game.events.on('chopshop:sold', ({ amount }) => {
        this.fromChopShop += amount;
        this.publish(0);
      }),
    );
  }

  dispose(): void {
    for (const u of this.unsubs) u();
    this.unsubs.length = 0;
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
    this.game.events.emit('economy:cashChanged', {
      cash: this.balance,
      delta: rounded,
      reason,
    });
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

  /**
   * Charges a fraction of the current balance (floored at `min`, capped at the
   * balance). Used for death/arrest bills; returns the amount actually taken.
   */
  fee(fraction: number, reason: string, min = 0): number {
    const wanted = Math.max(min, Math.round(this.balance * fraction));
    const taken = Math.min(this.balance, wanted);
    if (taken > 0) this.add(-taken, reason);
    return taken;
  }

  private publish(delta: number): void {
    const s = this.game.state;
    s.cash = this.balance;
    s.cashDelta = delta;
    s.stats.cashEarned = this.earned;
    s.stats.cashSpent = this.spent;
    s.stats.moneyFromRobberies = this.fromRobberies;
    s.stats.moneyFromChopShop = this.fromChopShop;
  }

  serialize(): unknown {
    const data: EconomySave = {
      cash: this.balance,
      earned: this.earned,
      spent: this.spent,
      fromRobberies: this.fromRobberies,
      fromChopShop: this.fromChopShop,
    };
    return data;
  }

  deserialize(data: unknown): void {
    const d = data as EconomySave | undefined;
    if (typeof d?.cash === 'number') this.balance = d.cash;
    if (typeof d?.earned === 'number') this.earned = d.earned;
    if (typeof d?.spent === 'number') this.spent = d.spent;
    if (typeof d?.fromRobberies === 'number') this.fromRobberies = d.fromRobberies;
    if (typeof d?.fromChopShop === 'number') this.fromChopShop = d.fromChopShop;
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
    fee: (fraction: unknown, reason: unknown = 'debug'): number => this.fee(Number(fraction), String(reason)),
    stats: (): {
      earned: number;
      spent: number;
      fromRobberies: number;
      fromChopShop: number;
    } => ({
      earned: this.earned,
      spent: this.spent,
      fromRobberies: this.fromRobberies,
      fromChopShop: this.fromChopShop,
    }),
  };
}

const module: SystemModule = {
  id: 'economy',
  priority: PRIORITY.economy,
  create: (game) => new EconomySystem(game),
};

export default module;
