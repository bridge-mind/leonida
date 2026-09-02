import { evaluate, SLOT_BETS, SLOT_PAYTABLE, SLOT_SYMBOLS, spin as spinReels, type SlotResult, type SlotSymbol } from '../../casino/Slots';
import type { CasinoSystem } from '../../systems/casino';
import { CasinoConfig } from '../../systems/casino/config';
import { CasinoPanel } from './Panel';

/** Glyphs drawn on the HTML reels (system fonts only, so no emoji). */
export const SLOT_GLYPHS: Record<SlotSymbol, string> = {
  cherry: '●●',
  bar: 'BAR',
  seven: '7',
  diamond: '◆',
  gator: 'GATOR',
};

/** Three-reel slot machine. Reels cycle symbols while spinning and stop staggered. */
export class SlotsPanel extends CasinoPanel {
  private bet: number = SLOT_BETS[0];
  private spinning = false;
  private machine = 0;
  private readonly reelEls: HTMLElement[] = [];
  private betBtns: HTMLButtonElement[] = [];
  private spinBtn!: HTMLButtonElement;
  private payoutEl!: HTMLElement;
  private raf = 0;
  private loop: ReturnType<CasinoSystem['game']['audio']['loop']> | null = null;
  lastResult: SlotResult | null = null;

  constructor(casino: CasinoSystem) {
    super(casino, 'slots', 'Slots');
    this.machine = casino.stage.nearestSlot();
  }

  protected build(): void {
    const cabinet = this.div('sl-cabinet');
    const reels = this.div('sl-reels');
    for (let i = 0; i < 3; i++) {
      const reel = this.div('sl-reel');
      reel.dataset.reel = String(i);
      this.setReel(reel, SLOT_SYMBOLS[(i * 2) % SLOT_SYMBOLS.length]);
      reels.appendChild(reel);
      this.reelEls.push(reel);
    }
    this.payoutEl = this.div('casino-msg');
    this.payoutEl.dataset.field = 'payout';
    cabinet.append(reels, this.buildPaytable());
    const betRow = this.row('Bet');
    for (const b of SLOT_BETS) {
      const btn = this.button(String(b), () => this.setBet(b), 'secondary', `bet-${b}`);
      this.betBtns.push(btn);
      betRow.appendChild(btn);
    }
    this.spinBtn = this.button('Pull lever', () => this.spin(), '', 'spin');
    betRow.appendChild(this.spinBtn);
    this.body.append(cabinet, betRow, this.msgEl);
    this.setBet(this.bet);
  }

  private buildPaytable(): HTMLElement {
    const table = this.div('sl-paytable');
    const rows: [string, number][] = [
      ['◆ ◆ ◆ JACKPOT', SLOT_PAYTABLE.threeDiamonds],
      ['GATOR ×3', SLOT_PAYTABLE.threeGators],
      ['7 7 7', SLOT_PAYTABLE.threeSevens],
      ['BAR BAR BAR', SLOT_PAYTABLE.threeBars],
      ['●● ×3', SLOT_PAYTABLE.threeCherries],
      ['●● ×2', SLOT_PAYTABLE.twoCherries],
    ];
    for (const [label, mult] of rows) table.appendChild(this.div('', `${label}  ${mult}×`));
    return table;
  }

  protected override canClose(): boolean {
    return !this.spinning;
  }

  protected override onClosed(): void {
    cancelAnimationFrame(this.raf);
    if (this.loop) this.casino.game.audio.stop(this.loop);
    this.casino.stage.setSlotScreen(this.machine, this.lastResult?.reels ?? null, false);
  }

  setBet(bet: number): void {
    if (this.spinning || !SLOT_BETS.includes(bet)) return;
    this.bet = bet;
    for (const b of this.betBtns) b.classList.toggle('active', b.textContent === String(bet));
  }

  spin(): void {
    if (this.spinning) return;
    if (!this.casino.takeChips(this.bet)) return this.fail(`You need ${this.bet} chips to play.`);
    const rigged = this.casino.rig.takeSlots();
    const result = rigged ? evaluate(rigged, this.bet) : spinReels(this.bet, this.casino.rng);
    this.lastResult = result;
    this.spinning = true;
    this.spinBtn.disabled = true;
    this.message('');
    this.payoutEl.textContent = '';
    const audio = this.casino.game.audio;
    audio.play('casino.slot.lever');
    this.loop = audio.loop('casino.reel.spin.loop');
    this.casino.stage.setSlotScreen(this.machine, null, true);
    for (const el of this.reelEls) el.classList.add('spinning');
    const stops = CasinoConfig.slots.reelStops;
    const stopped = [false, false, false];
    const start = performance.now();
    let lastCycle = 0;
    const tick = (): void => {
      const now = performance.now();
      const t = (now - start) / 1000;
      if (now - lastCycle >= 1000 / CasinoConfig.slots.reelRate) {
        lastCycle = now;
        for (let i = 0; i < 3; i++) {
          if (!stopped[i]) this.setReel(this.reelEls[i], SLOT_SYMBOLS[Math.floor(this.casino.rng() * SLOT_SYMBOLS.length)]);
        }
      }
      let allStopped = true;
      for (let i = 0; i < 3; i++) {
        if (!stopped[i] && t >= stops[i]) {
          stopped[i] = true;
          this.reelEls[i].classList.remove('spinning');
          this.setReel(this.reelEls[i], result.reels[i]);
          audio.play('casino.reel.stop');
        }
        if (!stopped[i]) allStopped = false;
      }
      if (allStopped) this.finish(result);
      else this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  private finish(result: SlotResult): void {
    const audio = this.casino.game.audio;
    if (this.loop) audio.stop(this.loop);
    this.loop = null;
    this.casino.stage.setSlotScreen(this.machine, result.reels, false);
    this.casino.giveChips(result.payout);
    this.casino.settleRound('slots', result.payout - result.bet);
    if (result.jackpot) {
      audio.play('casino.jackpot');
      this.message(`JACKPOT! ${result.payout.toLocaleString('en-US')} chips!`, 'win');
    } else if (result.payout > 0) {
      audio.play('casino.slot.win');
      audio.play('casino.slot.coin');
      this.message(`${result.label} — ${result.payout.toLocaleString('en-US')} chips`, 'win');
    } else {
      audio.play('casino.lose');
      this.message('No luck. Pull again?', 'lose');
    }
    this.payoutEl.textContent = String(result.payout);
    this.spinning = false;
    this.spinBtn.disabled = false;
  }

  private setReel(el: HTMLElement, symbol: SlotSymbol): void {
    el.textContent = SLOT_GLYPHS[symbol];
    el.className = `sl-reel sl-${symbol}${el.classList.contains('spinning') ? ' spinning' : ''}`;
    el.dataset.symbol = symbol;
  }
}
