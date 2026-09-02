/**
 * Playing cards and the multi-deck shoe shared by the blackjack logic.
 * Pure: no Three/DOM, randomness is injected as `rng: () => number` in [0, 1).
 */
export const SUITS = ['S', 'H', 'D', 'C'] as const;
export const RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'] as const;

export type Suit = (typeof SUITS)[number];
export type Rank = (typeof RANKS)[number];

export interface Card {
  readonly rank: Rank;
  readonly suit: Suit;
}

export interface HandValue {
  /** Best total ≤ 21 when possible, otherwise the minimum (busted) total. */
  total: number;
  /** True when an ace is currently counted as 11. */
  soft: boolean;
}

const SUIT_SET: ReadonlySet<string> = new Set(SUITS);
const RANK_SET: ReadonlySet<string> = new Set(RANKS);

/** Parses codes such as `AS`, `10H`, `TH`, `kd` (case-insensitive). Throws on malformed input. */
export function parseCard(code: string): Card {
  const text = code.trim().toUpperCase();
  const suit = text.slice(-1);
  let rank = text.slice(0, -1);
  if (rank === 'T') rank = '10';
  if (!SUIT_SET.has(suit) || !RANK_SET.has(rank)) throw new Error(`bad card code '${code}'`);
  return { rank: rank as Rank, suit: suit as Suit };
}

export function cardCode(card: Card): string {
  return `${card.rank}${card.suit}`;
}

export function isRed(card: Card): boolean {
  return card.suit === 'H' || card.suit === 'D';
}

export const SUIT_SYMBOL: Record<Suit, string> = { S: '\u2660', H: '\u2665', D: '\u2666', C: '\u2663' };

/** Blackjack value of a rank; aces count 11 here and are demoted by `handValue`. */
export function rankValue(rank: Rank): number {
  if (rank === 'A') return 11;
  if (rank === 'K' || rank === 'Q' || rank === 'J') return 10;
  return Number(rank);
}

export function handValue(cards: readonly Card[], out: HandValue = { total: 0, soft: false }): HandValue {
  let total = 0;
  let aces = 0;
  for (let i = 0; i < cards.length; i++) {
    const v = rankValue(cards[i].rank);
    total += v;
    if (v === 11) aces++;
  }
  while (total > 21 && aces > 0) {
    total -= 10;
    aces--;
  }
  out.total = total;
  out.soft = aces > 0;
  return out;
}

export function isTwentyOne(cards: readonly Card[]): boolean {
  return handValue(cards).total === 21;
}

/**
 * Multi-deck shoe. `force()` queues cards that are dealt before the shuffled
 * stock, which is how debug rigs and tests script exact hands.
 */
export class Shoe {
  private readonly stock: Card[] = [];
  private readonly forced: Card[] = [];
  /** Set when the last `draw()` had to reshuffle because the stock ran dry mid-hand. */
  shuffledMidHand = false;

  constructor(
    readonly decks: number,
    private readonly rng: () => number,
    /** Reshuffle before the next deal once this fraction (or less) of the shoe remains. */
    readonly reshuffleFraction = 0.25,
  ) {
    this.shuffle();
  }

  get total(): number {
    return this.decks * 52;
  }

  get remaining(): number {
    return this.stock.length + this.forced.length;
  }

  get needsShuffle(): boolean {
    return this.stock.length <= this.total * this.reshuffleFraction;
  }

  shuffle(): void {
    const stock = this.stock;
    stock.length = 0;
    for (let d = 0; d < this.decks; d++) {
      for (const suit of SUITS) for (const rank of RANKS) stock.push({ rank, suit });
    }
    for (let i = stock.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1));
      const tmp = stock[i];
      stock[i] = stock[j];
      stock[j] = tmp;
    }
  }

  /** Queues cards (codes or Card objects) to be drawn next, in order. */
  force(cards: readonly (string | Card)[]): void {
    this.forced.length = 0;
    for (const c of cards) this.forced.push(typeof c === 'string' ? parseCard(c) : c);
  }

  clearForced(): void {
    this.forced.length = 0;
  }

  draw(): Card {
    const forced = this.forced.shift();
    if (forced) return forced;
    if (this.stock.length === 0) {
      this.shuffle();
      this.shuffledMidHand = true;
    }
    return this.stock.pop() as Card;
  }
}
