import type { Blackjack, BlackjackHand, HandOutcome } from '../../casino/Blackjack';
import { SUIT_SYMBOL, handValue, isRed, rankValue, type Card } from '../../casino/cards';
import type { CasinoSystem } from '../../systems/casino';
import { CasinoPanel } from './Panel';

const DEAL_STEP_MS = 260;
const OUTCOME_TEXT: Record<HandOutcome, string> = {
  blackjack: 'BLACKJACK! Pays 3 to 2',
  win: 'You win',
  push: 'Push',
  lose: 'Dealer wins',
  bust: 'Bust',
};

/**
 * HTML blackjack table over the pure `Blackjack` logic. Cards animate in one at a
 * time; actions are disabled while cards are moving.
 */
export class BlackjackPanel extends CasinoPanel {
  private bet: number;
  private betInput!: HTMLInputElement;
  private dealBtn!: HTMLButtonElement;
  private actionBtns!: Record<'hit' | 'stand' | 'double' | 'split' | 'newHand', HTMLButtonElement>;
  private dealerCaption!: HTMLElement;
  private dealerHand!: HTMLElement;
  private seats!: HTMLElement;
  private readonly seatEls: { root: HTMLElement; caption: HTMLElement; hand: HTMLElement }[] = [];
  private animating = false;
  private holeEl: HTMLElement | null = null;
  private settledMessage: string | null = null;

  constructor(
    casino: CasinoSystem,
    readonly table: Blackjack,
    tier: 'low' | 'high',
  ) {
    super(casino, `blackjack-${tier}`, `Blackjack  $${table.limits.min.toLocaleString('en-US')} – $${table.limits.max.toLocaleString('en-US')}`);
    this.bet = table.limits.min;
  }

  protected build(): void {
    const felt = this.div('bj-table');
    this.dealerCaption = this.div('bj-caption', 'Dealer');
    this.dealerHand = this.div('bj-hand');
    this.seats = this.div('bj-seats');
    felt.append(this.dealerCaption, this.dealerHand, this.seats);

    this.betInput = document.createElement('input');
    this.betInput.type = 'number';
    this.betInput.className = 'casino-input';
    this.betInput.min = String(this.table.limits.min);
    this.betInput.max = String(this.table.limits.max);
    this.betInput.step = '1';
    this.betInput.value = String(this.bet);
    this.betInput.dataset.field = 'bet';
    this.betInput.addEventListener('change', () => this.setBet(Number(this.betInput.value)));
    const betRow = this.row('Bet', this.betInput);
    for (const amount of this.betPresets()) {
      betRow.appendChild(this.button(amount.toLocaleString('en-US'), () => this.setBet(amount), 'secondary small'));
    }
    this.dealBtn = this.button('Deal', () => this.deal(), '', 'deal');
    betRow.appendChild(this.dealBtn);

    this.actionBtns = {
      hit: this.button('Hit', () => this.hit(), '', 'hit'),
      stand: this.button('Stand', () => this.stand(), '', 'stand'),
      double: this.button('Double', () => this.double(), '', 'double'),
      split: this.button('Split', () => this.split(), '', 'split'),
      newHand: this.button('New hand', () => this.newHand(), 'secondary', 'new-hand'),
    };
    const actions = this.row(this.actionBtns.hit, this.actionBtns.stand, this.actionBtns.double, this.actionBtns.split, this.actionBtns.newHand);
    this.body.append(felt, betRow, actions, this.msgEl);
    // A round may be mid-flight if the player walked away and came back.
    this.syncAll(false);
    this.refreshButtons();
  }

  private betPresets(): number[] {
    const { min, max } = this.table.limits;
    const set = new Set<number>([min, min * 2, min * 5, min * 10, max]);
    return [...set].filter((v) => v >= min && v <= max).sort((a, b) => a - b);
  }

  protected override canClose(): boolean {
    return !this.animating;
  }

  /** Read through a call so TypeScript does not narrow `phase` across `deal()`. */
  private isSettled(): boolean {
    return this.table.state.phase === 'settled';
  }

  // --- Public actions (also driven by the autotest) -----------------------------

  setBet(amount: number): void {
    const { min, max } = this.table.limits;
    this.bet = Math.min(max, Math.max(min, Math.round(Number.isFinite(amount) ? amount : min)));
    this.betInput.value = String(this.bet);
  }

  deal(): void {
    if (this.animating || this.table.state.phase !== 'betting') return;
    if (!this.table.isValidBet(this.bet)) return this.fail(`Bet must be between ${this.table.limits.min} and ${this.table.limits.max}.`);
    if (!this.casino.takeChips(this.bet)) return this.fail('Not enough chips. Visit the cashier.');
    const rigged = this.casino.rig.takeBlackjack();
    if (rigged) this.table.shoe.force(rigged);
    const { shuffled } = this.table.deal(this.bet);
    if (shuffled) this.casino.game.audio.play('casino.card.shuffle');
    this.casino.game.audio.play('casino.chip.place');
    this.message('');
    this.settledMessage = null;
    this.clearTable();
    this.buildSeats();
    // Deal order: player, dealer, player, dealer (hole card face down).
    const hand = this.table.state.hands[0];
    const dealer = this.table.state.dealer;
    // A natural on either side settles immediately, so the hole card is shown from the start.
    const holeHidden = !this.isSettled();
    this.animating = true;
    this.refreshButtons();
    this.after(0, () => this.addCard(this.seatEls[0].hand, hand.cards[0]));
    this.after(DEAL_STEP_MS, () => this.addCard(this.dealerHand, dealer[0]));
    this.after(DEAL_STEP_MS * 2, () => this.addCard(this.seatEls[0].hand, hand.cards[1]));
    this.after(DEAL_STEP_MS * 3, () => {
      this.holeEl = this.addCard(this.dealerHand, dealer[1], holeHidden);
      this.updateCaptions();
    });
    this.after(DEAL_STEP_MS * 4, () => {
      this.animating = false;
      if (this.table.state.phase === 'settled') this.finish();
      else this.refreshButtons();
    });
  }

  hit(): void {
    if (this.animating || !this.table.canHit()) return;
    const idx = this.table.state.active;
    const card = this.table.hit();
    this.addCard(this.seatEls[idx].hand, card);
    this.updateCaptions();
    this.afterAction();
  }

  stand(): void {
    if (this.animating || !this.table.canStand()) return;
    this.table.stand();
    this.afterAction();
  }

  double(): void {
    if (this.animating || !this.table.canDouble()) return;
    const hand = this.table.activeHand;
    if (!this.casino.takeChips(hand.bet)) return this.fail('Not enough chips to double.');
    this.casino.game.audio.play('casino.chip.place');
    const idx = this.table.state.active;
    const card = this.table.double();
    this.addCard(this.seatEls[idx].hand, card);
    this.updateCaptions();
    this.afterAction();
  }

  split(): void {
    if (this.animating || !this.table.canSplit()) return;
    const hand = this.table.activeHand;
    if (!this.casino.takeChips(hand.bet)) return this.fail('Not enough chips to split.');
    this.casino.game.audio.play('casino.chip.place');
    this.table.split();
    this.clearSeats();
    this.buildSeats();
    this.syncAll(true);
    this.afterAction();
  }

  newHand(): void {
    if (this.animating || this.table.state.phase !== 'settled') return;
    this.table.reset();
    this.clearTable();
    this.message('');
    this.refreshButtons();
  }

  // --- Flow --------------------------------------------------------------------

  private afterAction(): void {
    this.updateCaptions();
    if (this.table.state.phase !== 'settled') {
      this.refreshButtons();
      return;
    }
    // Dealer reveal and draws are already decided; animate them one card at a time.
    this.animating = true;
    this.refreshButtons();
    const dealer = this.table.state.dealer;
    let delay = 0;
    if (this.holeEl) {
      this.after(delay, () => this.revealHole());
      delay += DEAL_STEP_MS * 1.5;
    }
    for (let i = 2; i < dealer.length; i++) {
      const card = dealer[i];
      this.after(delay, () => {
        this.addCard(this.dealerHand, card);
        this.updateCaptions();
      });
      delay += DEAL_STEP_MS * 1.5;
    }
    this.after(delay, () => {
      this.animating = false;
      this.finish();
    });
  }

  private finish(): void {
    if (this.settledMessage !== null) return;
    if (this.holeEl) this.revealHole();
    const res = this.table.resolve();
    let payout = 0;
    for (const h of res.hands) payout += h.payout;
    this.casino.giveChips(payout);
    this.casino.settleRound('blackjack', res.net);
    this.updateCaptions();
    const parts = res.hands.map((h, i) => (res.hands.length > 1 ? `Hand ${i + 1}: ${OUTCOME_TEXT[h.outcome]}` : OUTCOME_TEXT[h.outcome]));
    const delta = res.net > 0 ? ` (+${res.net.toLocaleString('en-US')})` : res.net < 0 ? ` (${res.net.toLocaleString('en-US')})` : ' (bet returned)';
    this.settledMessage = parts.join('  ·  ') + delta;
    this.message(this.settledMessage, res.net > 0 ? 'win' : res.net < 0 ? 'lose' : 'info');
    const best = res.hands.some((h) => h.outcome === 'blackjack')
      ? 'casino.blackjack'
      : res.net > 0
        ? 'casino.win'
        : res.net < 0
          ? res.hands.every((h) => h.outcome === 'bust')
            ? 'casino.bust'
            : 'casino.lose'
          : 'casino.push';
    this.casino.game.audio.play(best);
    if (payout > 0) this.casino.game.audio.play('casino.chip.collect');
    this.refreshButtons();
  }

  // --- DOM --------------------------------------------------------------------

  private clearTable(): void {
    this.dealerHand.replaceChildren();
    this.holeEl = null;
    this.clearSeats();
    this.dealerCaption.textContent = 'Dealer';
  }

  private clearSeats(): void {
    this.seats.replaceChildren();
    this.seatEls.length = 0;
  }

  private buildSeats(): void {
    for (let i = 0; i < this.table.state.hands.length; i++) {
      const root = this.div('bj-seat');
      const caption = this.div('bj-caption', 'You');
      const hand = this.div('bj-hand');
      root.append(caption, hand);
      this.seats.appendChild(root);
      this.seatEls.push({ root, caption, hand });
    }
  }

  /** Adds any cards the DOM does not show yet (used after a split and when re-opening). */
  private syncAll(animate: boolean): void {
    const s = this.table.state;
    if (s.phase === 'betting') return;
    if (this.seatEls.length !== s.hands.length) {
      this.clearSeats();
      this.buildSeats();
    }
    s.hands.forEach((h, i) => this.syncHand(this.seatEls[i].hand, h.cards, animate));
    if (this.dealerHand.childElementCount === 0) {
      this.addCard(this.dealerHand, s.dealer[0], false, animate);
      this.holeEl = s.dealer[1] ? this.addCard(this.dealerHand, s.dealer[1], s.holeHidden, animate) : null;
      for (let i = 2; i < s.dealer.length; i++) this.addCard(this.dealerHand, s.dealer[i], false, animate);
    }
    this.updateCaptions();
  }

  /** Closing mid-round keeps the hand; a finished round is paid (if the animation was cut) and cleared. */
  protected override onClosed(): void {
    if (this.table.state.phase !== 'settled') return;
    if (this.settledMessage === null) this.finish();
    this.table.reset();
  }

  private syncHand(container: HTMLElement, cards: readonly Card[], animate: boolean): void {
    container.replaceChildren();
    for (const c of cards) this.addCard(container, c, false, animate);
  }

  private addCard(container: HTMLElement, card: Card, faceDown = false, animate = true): HTMLElement {
    const el = document.createElement('div');
    el.className = `bj-card${isRed(card) ? ' red' : ''}${faceDown ? ' back' : ''}`;
    if (!animate) el.style.animation = 'none';
    el.dataset.card = `${card.rank}${card.suit}`;
    const top = this.div('bj-rank', card.rank);
    const suit = this.div('bj-suit', SUIT_SYMBOL[card.suit]);
    const bottom = this.div('bj-rank bj-rank-bottom', card.rank);
    el.append(top, suit, bottom);
    container.appendChild(el);
    this.casino.game.audio.play('casino.card.deal');
    return el;
  }

  private revealHole(): void {
    if (!this.holeEl) return;
    this.holeEl.classList.remove('back');
    this.holeEl.style.animation = 'none';
    this.holeEl = null;
    this.casino.game.audio.play('casino.card.flip');
    this.updateCaptions();
  }

  private updateCaptions(): void {
    const s = this.table.state;
    if (s.dealer.length === 0) {
      this.dealerCaption.textContent = 'Dealer';
    } else if (s.holeHidden || this.holeEl) {
      const up = s.dealer[0];
      this.dealerCaption.textContent = `Dealer shows ${up.rank === 'A' ? 'Ace' : rankValue(up.rank)}`;
    } else {
      const v = handValue(s.dealer);
      this.dealerCaption.textContent = v.total > 21 ? `Dealer busts with ${v.total}` : `Dealer has ${v.total}`;
    }
    s.hands.forEach((h, i) => {
      const seat = this.seatEls[i];
      if (!seat) return;
      seat.caption.textContent = this.handCaption(h, s.hands.length > 1 ? i + 1 : 0);
      seat.root.classList.toggle('active', s.phase === 'player' && s.active === i);
    });
  }

  private handCaption(h: BlackjackHand, index: number): string {
    const v = handValue(h.cards);
    const name = index ? `Hand ${index}` : 'You';
    const soft = v.soft && v.total <= 21 ? 'soft ' : '';
    const flags = `${h.doubled ? ' · doubled' : ''}${h.natural ? ' · blackjack' : ''}`;
    return `${name}: ${soft}${v.total}${v.total > 21 ? ' (bust)' : ''} · bet ${h.bet.toLocaleString('en-US')}${flags}`;
  }

  private refreshButtons(): void {
    const t = this.table;
    const idle = !this.animating;
    const betting = t.state.phase === 'betting';
    this.dealBtn.disabled = !idle || !betting;
    this.betInput.disabled = !betting;
    const canDouble = idle && t.canDouble() && this.casino.chips >= t.activeHand.bet;
    const canSplit = idle && t.canSplit() && this.casino.chips >= t.activeHand.bet;
    this.actionBtns.hit.disabled = !idle || !t.canHit();
    this.actionBtns.stand.disabled = !idle || !t.canStand();
    this.actionBtns.double.disabled = !canDouble;
    this.actionBtns.split.disabled = !canSplit;
    this.actionBtns.newHand.disabled = !idle || t.state.phase !== 'settled';
  }
}
