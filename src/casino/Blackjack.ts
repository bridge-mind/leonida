/**
 * Blackjack table logic (SYSTEM_PROMPT §11). Pure and deterministic given the shoe.
 *
 * House rules: 6-deck shoe reshuffled at 25% penetration, dealer stands on soft 17,
 * natural blackjack pays 3:2, push returns the bet, no insurance, double on the
 * first two cards only (one card then stand), one split of a pair (no re-split),
 * split aces receive exactly one card, 21 after a split is a plain win.
 *
 * Chips flow: the caller deducts each bet when it is placed (`deal`, `double`, `split`)
 * and credits `resolve().hands[i].payout` afterwards; `net` is the round's delta.
 */
import { Shoe, handValue, rankValue, type Card, type HandValue } from './cards';

export interface TableLimits {
  min: number;
  max: number;
}

export type BlackjackPhase = 'betting' | 'player' | 'settled';
export type HandOutcome = 'blackjack' | 'win' | 'push' | 'lose' | 'bust';

export interface BlackjackHand {
  cards: Card[];
  bet: number;
  doubled: boolean;
  fromSplit: boolean;
  /** Player can no longer act on this hand. */
  done: boolean;
  /** Two-card 21 dealt from the initial deal (not after a split). */
  natural: boolean;
}

export interface HandResult {
  outcome: HandOutcome;
  bet: number;
  /** Chips returned to the player for this hand (0 when lost). */
  payout: number;
  total: number;
}

export interface BlackjackResolution {
  hands: HandResult[];
  dealerTotal: number;
  /** Total payout minus total wagered this round. */
  net: number;
}

export interface BlackjackState {
  phase: BlackjackPhase;
  hands: BlackjackHand[];
  /** Index of the hand the player is acting on. */
  active: number;
  dealer: Card[];
  /** Dealer's second card is face down until the player finishes. */
  holeHidden: boolean;
}

export const BLACKJACK_DECKS = 6;
export const BLACKJACK_RESHUFFLE_FRACTION = 0.25;
export const BLACKJACK_PAYS = 1.5;
const DEALER_STANDS_AT = 17;

export class Blackjack {
  readonly shoe: Shoe;
  readonly state: BlackjackState = { phase: 'betting', hands: [], active: 0, dealer: [], holeHidden: true };
  private readonly scratch: HandValue = { total: 0, soft: false };
  private resolution: BlackjackResolution | null = null;

  constructor(
    readonly limits: TableLimits,
    rng: () => number,
    shoe?: Shoe,
  ) {
    this.shoe = shoe ?? new Shoe(BLACKJACK_DECKS, rng, BLACKJACK_RESHUFFLE_FRACTION);
  }

  isValidBet(bet: number): boolean {
    return Number.isInteger(bet) && bet >= this.limits.min && bet <= this.limits.max;
  }

  get activeHand(): BlackjackHand {
    return this.state.hands[this.state.active];
  }

  dealerUpCard(): Card {
    return this.state.dealer[0];
  }

  /** Total wagered in the current round (all hands, including doubles). */
  get wagered(): number {
    let sum = 0;
    for (const h of this.state.hands) sum += h.bet;
    return sum;
  }

  /** Starts a round. Returns whether the shoe was reshuffled first (for the shuffle sound). */
  deal(bet: number): { shuffled: boolean } {
    if (this.state.phase !== 'betting') throw new Error('blackjack: round in progress');
    if (!this.isValidBet(bet)) throw new Error(`blackjack: bet ${bet} outside limits ${this.limits.min}-${this.limits.max}`);
    let shuffled = false;
    if (this.shoe.needsShuffle) {
      this.shoe.shuffle();
      shuffled = true;
    }
    this.shoe.shuffledMidHand = false;
    this.resolution = null;
    const s = this.state;
    s.hands.length = 0;
    s.dealer.length = 0;
    s.active = 0;
    s.holeHidden = true;
    const hand: BlackjackHand = { cards: [], bet, doubled: false, fromSplit: false, done: false, natural: false };
    s.hands.push(hand);
    hand.cards.push(this.shoe.draw());
    s.dealer.push(this.shoe.draw());
    hand.cards.push(this.shoe.draw());
    s.dealer.push(this.shoe.draw());

    const playerNatural = handValue(hand.cards, this.scratch).total === 21;
    const dealerNatural = handValue(s.dealer, this.scratch).total === 21;
    if (playerNatural || dealerNatural) {
      hand.natural = playerNatural;
      hand.done = true;
      s.holeHidden = false;
      s.phase = 'settled';
    } else {
      s.phase = 'player';
    }
    return { shuffled };
  }

  // --- Action availability -----------------------------------------------------

  canHit(): boolean {
    return this.state.phase === 'player' && !this.activeHand.done;
  }

  canStand(): boolean {
    return this.canHit();
  }

  canDouble(): boolean {
    if (!this.canHit()) return false;
    const h = this.activeHand;
    return h.cards.length === 2 && !h.doubled;
  }

  canSplit(): boolean {
    if (!this.canHit()) return false;
    const h = this.activeHand;
    if (h.fromSplit || h.cards.length !== 2 || this.state.hands.length > 1) return false;
    return rankValue(h.cards[0].rank) === rankValue(h.cards[1].rank);
  }

  // --- Actions -------------------------------------------------------------------

  hit(): Card {
    if (!this.canHit()) throw new Error('blackjack: cannot hit');
    const h = this.activeHand;
    const card = this.shoe.draw();
    h.cards.push(card);
    const total = handValue(h.cards, this.scratch).total;
    if (total >= 21) h.done = true;
    this.advance();
    return card;
  }

  stand(): void {
    if (!this.canStand()) throw new Error('blackjack: cannot stand');
    this.activeHand.done = true;
    this.advance();
  }

  /** Doubles the bet (caller deducts `hand.bet` more chips), takes exactly one card, stands. */
  double(): Card {
    if (!this.canDouble()) throw new Error('blackjack: cannot double');
    const h = this.activeHand;
    h.bet *= 2;
    h.doubled = true;
    const card = this.shoe.draw();
    h.cards.push(card);
    h.done = true;
    this.advance();
    return card;
  }

  /** Splits the pair into two hands (caller deducts one more bet). Split aces get one card each and stand. */
  split(): void {
    if (!this.canSplit()) throw new Error('blackjack: cannot split');
    const s = this.state;
    const first = this.activeHand;
    const moved = first.cards.pop() as Card;
    const second: BlackjackHand = { cards: [moved], bet: first.bet, doubled: false, fromSplit: true, done: false, natural: false };
    first.fromSplit = true;
    s.hands.push(second);
    first.cards.push(this.shoe.draw());
    second.cards.push(this.shoe.draw());
    const aces = moved.rank === 'A';
    if (aces) {
      first.done = true;
      second.done = true;
    } else {
      if (handValue(first.cards, this.scratch).total === 21) first.done = true;
      if (handValue(second.cards, this.scratch).total === 21) second.done = true;
    }
    this.advance();
  }

  /** Moves to the next unfinished hand or lets the dealer play. */
  private advance(): void {
    const s = this.state;
    while (s.active < s.hands.length && s.hands[s.active].done) s.active++;
    if (s.active < s.hands.length) return;
    s.active = s.hands.length - 1;
    this.playDealer();
  }

  private playDealer(): void {
    const s = this.state;
    s.holeHidden = false;
    let anyLive = false;
    for (const h of s.hands) if (handValue(h.cards, this.scratch).total <= 21) anyLive = true;
    // With every player hand busted the dealer just reveals; drawing would only burn cards.
    if (anyLive) {
      while (handValue(s.dealer, this.scratch).total < DEALER_STANDS_AT) s.dealer.push(this.shoe.draw());
    }
    s.phase = 'settled';
  }

  /** Per-hand outcomes and the net chip delta. Only valid once `phase === 'settled'`. */
  resolve(): BlackjackResolution {
    if (this.state.phase !== 'settled') throw new Error('blackjack: round not finished');
    if (this.resolution) return this.resolution;
    const dealerTotal = handValue(this.state.dealer, this.scratch).total;
    const dealerNatural = this.state.dealer.length === 2 && dealerTotal === 21;
    const hands: HandResult[] = [];
    let net = 0;
    for (const h of this.state.hands) {
      const total = handValue(h.cards, this.scratch).total;
      let outcome: HandOutcome;
      if (total > 21) outcome = 'bust';
      else if (h.natural) outcome = dealerNatural ? 'push' : 'blackjack';
      else if (dealerNatural) outcome = 'lose';
      else if (dealerTotal > 21 || total > dealerTotal) outcome = 'win';
      else if (total === dealerTotal) outcome = 'push';
      else outcome = 'lose';
      const payout = payoutFor(outcome, h.bet);
      net += payout - h.bet;
      hands.push({ outcome, bet: h.bet, payout, total });
    }
    this.resolution = { hands, dealerTotal, net };
    return this.resolution;
  }

  /** Clears the table for the next bet. */
  reset(): void {
    const s = this.state;
    s.phase = 'betting';
    s.hands.length = 0;
    s.dealer.length = 0;
    s.active = 0;
    s.holeHidden = true;
    this.resolution = null;
  }
}

export function payoutFor(outcome: HandOutcome, bet: number): number {
  switch (outcome) {
    case 'blackjack':
      return bet + Math.floor(bet * BLACKJACK_PAYS);
    case 'win':
      return bet * 2;
    case 'push':
      return bet;
    case 'lose':
    case 'bust':
      return 0;
  }
}
