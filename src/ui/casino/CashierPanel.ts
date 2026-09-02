import { CasinoConfig } from '../../systems/casino/config';
import type { CasinoSystem } from '../../systems/casino';
import { CasinoPanel } from './Panel';

/** Buy chips with cash (1:1) or cash chips out. */
export class CashierPanel extends CasinoPanel {
  private buyInput!: HTMLInputElement;
  private outInput!: HTMLInputElement;

  constructor(casino: CasinoSystem) {
    super(casino, 'cashier', 'Cashier');
  }

  protected build(): void {
    const economy = this.casino.game.economy;
    const grid = this.div('cs-grid');

    const buy = this.div('cs-box');
    buy.appendChild(Object.assign(document.createElement('h3'), { textContent: 'Buy chips' }));
    const presets = this.row('Presets');
    for (const amount of CasinoConfig.cashier.presets) {
      presets.appendChild(this.button(`$${amount.toLocaleString('en-US')}`, () => this.buy(amount), 'secondary', `buy-${amount}`));
    }
    buy.appendChild(presets);
    this.buyInput = this.input(1000);
    this.buyInput.dataset.field = 'buy';
    buy.appendChild(this.row(this.buyInput, this.button('Buy', () => this.buy(Number(this.buyInput.value)), '', 'buy')));
    buy.appendChild(this.div('casino-hint', 'Chips are 1:1 with cash. Table limits apply per game.'));

    const out = this.div('cs-box');
    out.appendChild(Object.assign(document.createElement('h3'), { textContent: 'Cash out' }));
    this.outInput = this.input(Math.max(0, this.casino.chips));
    this.outInput.dataset.field = 'cashout';
    out.appendChild(
      this.row(
        this.outInput,
        this.button('Cash out', () => this.cashOut(Number(this.outInput.value)), '', 'cashout'),
        this.button('All', () => this.cashOut(this.casino.chips), 'secondary', 'cashout-all'),
      ),
    );
    out.appendChild(this.div('casino-hint', 'Winnings go straight to your wallet.'));

    grid.append(buy, out);
    this.body.append(grid, this.msgEl);
    if (!economy) this.fail('The cashier is closed (economy system not loaded).');
  }

  private input(value: number): HTMLInputElement {
    const el = document.createElement('input');
    el.type = 'number';
    el.min = '1';
    el.step = '1';
    el.className = 'casino-input';
    el.value = String(Math.round(value));
    return el;
  }

  private buy(amount: number): void {
    if (!Number.isFinite(amount) || amount <= 0) return this.fail('Enter an amount.');
    if (this.casino.buyChips(amount)) {
      this.message(`Bought ${Math.round(amount).toLocaleString('en-US')} chips.`, 'win');
      this.casino.game.audio.play('casino.chip.stack');
    } else this.fail(this.casino.game.economy ? 'Not enough cash.' : 'Cashier closed.');
    this.outInput.value = String(this.casino.chips);
  }

  private cashOut(amount: number): void {
    if (!Number.isFinite(amount) || amount <= 0) return this.fail('Enter an amount.');
    if (this.casino.cashOut(amount)) {
      this.message(`Cashed out $${Math.round(amount).toLocaleString('en-US')}.`, 'win');
      this.casino.game.audio.play('casino.chip.collect');
    } else this.fail('Not enough chips.');
    this.outInput.value = String(this.casino.chips);
  }
}
