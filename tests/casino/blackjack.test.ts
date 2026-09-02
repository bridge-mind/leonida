import { describe, expect, it } from 'vitest';
import { Blackjack, payoutFor } from '../../src/casino/Blackjack';
import { Shoe, handValue, parseCard } from '../../src/casino/cards';
import { Random } from '../../src/core/Random';

const LIMITS = { min: 50, max: 2000 };

/** Table whose next draws are scripted: player, dealer, player, dealer, then hits in order. */
function rigged(cards: string[], seed = 1): Blackjack {
  const rng = new Random(seed);
  const table = new Blackjack(LIMITS, () => rng.next());
  table.shoe.force(cards);
  return table;
}

describe('cards', () => {
  it('parses codes and values hands with soft aces', () => {
    expect(parseCard('10H')).toEqual({ rank: '10', suit: 'H' });
    expect(parseCard('th')).toEqual({ rank: '10', suit: 'H' });
    expect(() => parseCard('1X')).toThrow();
    expect(handValue(['AS', 'KH'].map(parseCard))).toEqual({ total: 21, soft: true });
    expect(handValue(['AS', 'AH', '9D'].map(parseCard))).toEqual({ total: 21, soft: true });
    expect(handValue(['AS', '6H', 'KD'].map(parseCard))).toEqual({ total: 17, soft: false });
    expect(handValue(['KS', 'QH', '5D'].map(parseCard)).total).toBe(25);
  });

  it('shoe holds 6 decks and reshuffles at 25% penetration', () => {
    const rng = new Random(7);
    const shoe = new Shoe(6, () => rng.next());
    expect(shoe.total).toBe(312);
    expect(shoe.remaining).toBe(312);
    const seen = new Map<string, number>();
    for (let i = 0; i < 312; i++) {
      if (shoe.remaining === 78) expect(shoe.needsShuffle).toBe(true);
      const c = shoe.draw();
      seen.set(`${c.rank}${c.suit}`, (seen.get(`${c.rank}${c.suit}`) ?? 0) + 1);
    }
    expect(seen.size).toBe(52);
    for (const n of seen.values()) expect(n).toBe(6);
    expect(shoe.remaining).toBe(0);
    shoe.draw();
    expect(shoe.shuffledMidHand).toBe(true);
    expect(shoe.remaining).toBe(311);
  });
});

describe('Blackjack', () => {
  it('rejects bets outside the table limits and validates phases', () => {
    const table = rigged(['5S', '6H', '7D', '8C']);
    expect(table.isValidBet(49)).toBe(false);
    expect(table.isValidBet(2001)).toBe(false);
    expect(table.isValidBet(50.5)).toBe(false);
    expect(table.isValidBet(50)).toBe(true);
    expect(() => table.deal(10)).toThrow();
    expect(() => table.resolve()).toThrow();
    table.deal(100);
    expect(() => table.deal(100)).toThrow('round in progress');
  });

  it('natural blackjack pays 3:2 and settles immediately', () => {
    const table = rigged(['AS', '9D', 'KH', '7C']);
    table.deal(100);
    expect(table.state.phase).toBe('settled');
    expect(table.state.holeHidden).toBe(false);
    const res = table.resolve();
    expect(res.hands[0].outcome).toBe('blackjack');
    expect(res.hands[0].payout).toBe(250);
    expect(res.net).toBe(150);
    expect(payoutFor('blackjack', 75)).toBe(75 + 112);
  });

  it('player natural vs dealer natural is a push; dealer natural alone loses', () => {
    const push = rigged(['AS', 'AD', 'KH', 'KC']);
    push.deal(200);
    expect(push.resolve().hands[0].outcome).toBe('push');
    expect(push.resolve().net).toBe(0);

    const lose = rigged(['9S', 'AD', '9H', 'KC']);
    lose.deal(200);
    expect(lose.state.phase).toBe('settled');
    expect(lose.resolve().hands[0].outcome).toBe('lose');
    expect(lose.resolve().net).toBe(-200);
  });

  it('hit, bust and stand flow; dealer draws to 17 and stands on soft 17', () => {
    // Player 10+6 hits a 9 -> bust; dealer never draws when everyone busted.
    const bust = rigged(['TS', '9D', '6H', '7C', '9S']);
    bust.deal(50);
    expect(bust.canHit()).toBe(true);
    bust.hit();
    expect(bust.state.phase).toBe('settled');
    expect(bust.state.dealer).toHaveLength(2);
    expect(bust.resolve().hands[0].outcome).toBe('bust');
    expect(bust.resolve().net).toBe(-50);

    // Dealer A+6 (soft 17) must stand; player 20 wins 1:1.
    const soft17 = rigged(['TS', 'AD', 'KH', '6C', '5S']);
    soft17.deal(100);
    soft17.stand();
    expect(soft17.state.dealer).toHaveLength(2);
    expect(soft17.resolve().dealerTotal).toBe(17);
    expect(soft17.resolve().hands[0].outcome).toBe('win');
    expect(soft17.resolve().net).toBe(100);

    // Dealer 10+6 draws: 16 -> +2 = 18 beats player 17.
    const draws = rigged(['TS', 'TD', '7H', '6C', '2S']);
    draws.deal(100);
    draws.stand();
    expect(draws.state.dealer).toHaveLength(3);
    expect(draws.resolve().dealerTotal).toBe(18);
    expect(draws.resolve().hands[0].outcome).toBe('lose');

    // Equal totals push and return the bet.
    const push = rigged(['TS', 'TD', '8H', '8C']);
    push.deal(100);
    push.stand();
    expect(push.resolve().hands[0].outcome).toBe('push');
    expect(push.resolve().hands[0].payout).toBe(100);
  });

  it('double is only allowed on the first two cards, takes one card, then stands', () => {
    const table = rigged(['5S', '9D', '6H', '7C', 'TS', '2D', '5C']);
    table.deal(100);
    expect(table.canDouble()).toBe(true);
    table.double();
    expect(table.activeHand.cards).toHaveLength(3);
    expect(table.activeHand.bet).toBe(200);
    expect(table.state.phase).toBe('settled');
    // Dealer 9+7 = 16 draws 2 -> 18; player 21 wins double stake.
    const res = table.resolve();
    expect(res.hands[0].outcome).toBe('win');
    expect(res.hands[0].payout).toBe(400);
    expect(res.net).toBe(200);

    const three = rigged(['2S', '9D', '3H', '7C', '4S']);
    three.deal(100);
    three.hit();
    expect(three.activeHand.cards).toHaveLength(3);
    expect(three.canDouble()).toBe(false);
    expect(() => three.double()).toThrow();
  });

  it('splits a pair once, refuses re-splits, and 21 after a split is not a natural', () => {
    // Player 8,8 vs dealer 9,7. Split draws: hand1 gets 8 (pair again), hand2 gets 3.
    const table = rigged(['8S', '9D', '8H', '7C', '8D', '3S', 'TS', 'TD', '9C']);
    table.deal(100);
    expect(table.canSplit()).toBe(true);
    table.split();
    expect(table.state.hands).toHaveLength(2);
    expect(table.wagered).toBe(200);
    expect(table.canSplit()).toBe(false);
    expect(() => table.split()).toThrow();
    // Hand 1: 8+8 hits 10 -> 26 bust. Hand 2: 8+3 hits 10 -> 21 (not natural).
    table.hit();
    expect(table.state.active).toBe(1);
    table.hit();
    expect(table.state.phase).toBe('settled');
    const res = table.resolve();
    expect(res.hands[0].outcome).toBe('bust');
    expect(res.hands[1].outcome).toBe('win');
    expect(res.hands[1].payout).toBe(200);
    expect(res.hands[1].total).toBe(21);
    expect(res.dealerTotal).toBe(25);
    expect(res.net).toBe(-100 + 100);
  });

  it('a two-card 21 on a split hand pays 1:1, not 3:2', () => {
    // Player K,K vs dealer 9,8. Split: hand1 gets A (21), hand2 gets 5.
    const table = rigged(['KS', '9D', 'KH', '8C', 'AS', '5S', '4D']);
    table.deal(100);
    table.split();
    expect(table.state.hands[0].done).toBe(true);
    expect(table.state.active).toBe(1);
    table.stand();
    const res = table.resolve();
    expect(res.hands[0].outcome).toBe('win');
    expect(res.hands[0].payout).toBe(200);
    expect(res.hands[1].outcome).toBe('lose');
    expect(res.net).toBe(0);
  });

  it('split aces receive exactly one card each and cannot be hit', () => {
    const table = rigged(['AS', '9D', 'AH', '7C', 'KS', '5D', '3C']);
    table.deal(100);
    table.split();
    expect(table.state.phase).toBe('settled');
    expect(table.state.hands[0].cards).toHaveLength(2);
    expect(table.state.hands[1].cards).toHaveLength(2);
    expect(table.canHit()).toBe(false);
    const res = table.resolve();
    // Dealer 9+7 = 16 draws 3 -> 19. Hand1 A+K = 21 wins 1:1, hand2 A+5 = 16 loses.
    expect(res.dealerTotal).toBe(19);
    expect(res.hands[0].outcome).toBe('win');
    expect(res.hands[0].payout).toBe(200);
    expect(res.hands[1].outcome).toBe('lose');
    expect(res.net).toBe(0);
  });

  it('non-pairs cannot split and reset returns the table to betting', () => {
    const table = rigged(['8S', '9D', '7H', '7C']);
    table.deal(100);
    expect(table.canSplit()).toBe(false);
    table.stand();
    table.reset();
    expect(table.state.phase).toBe('betting');
    expect(table.state.hands).toHaveLength(0);
    expect(table.state.dealer).toHaveLength(0);
  });

  it('plays 200 random rounds without invalid states and with sane payouts', () => {
    const rng = new Random(2026);
    const table = new Blackjack(LIMITS, () => rng.next());
    let bank = 0;
    for (let round = 0; round < 200; round++) {
      table.reset();
      table.deal(50);
      while (table.state.phase === 'player') {
        if (table.canSplit() && rng.chance(0.5)) table.split();
        else if (table.canDouble() && rng.chance(0.2)) table.double();
        else if (handValue(table.activeHand.cards).total < 17) table.hit();
        else table.stand();
      }
      const res = table.resolve();
      expect(res.hands.length).toBeGreaterThan(0);
      for (const h of res.hands) {
        expect(h.payout).toBeGreaterThanOrEqual(0);
        expect(h.payout).toBeLessThanOrEqual(h.bet * 2.5);
      }
      // The dealer only draws when a non-natural player hand is still live.
      const dealerPlayed = res.hands.some((h) => h.outcome !== 'bust' && h.outcome !== 'blackjack');
      if (dealerPlayed) expect(res.dealerTotal).toBeGreaterThanOrEqual(17);
      bank += res.net;
    }
    // Basic strategy-ish play over 200 hands stays within a plausible band of the stake.
    expect(Math.abs(bank)).toBeLessThan(200 * 50);
  });
});
