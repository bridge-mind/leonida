/**
 * Three-reel slot machine with weighted symbols and a single centre payline.
 * Pure logic; `spin` takes an injected rng so tests and debug rigs are deterministic.
 *
 * Each reel has its own symbol weights (like real reel strips). With five always-paying
 * symbols and identical reels the return-to-player cannot drop below ~145%, so the reels
 * differ and a lone cherry pays nothing; the resulting RTP is ~92%.
 */
export const SLOT_SYMBOLS = ['cherry', 'bar', 'seven', 'diamond', 'gator'] as const;
export type SlotSymbol = (typeof SLOT_SYMBOLS)[number];

export const SLOT_BETS: readonly number[] = [10, 50, 100];

/** Relative frequency of each symbol per reel (index 0..2). */
export const REEL_WEIGHTS: readonly Record<SlotSymbol, number>[] = [
  { cherry: 3, bar: 8, seven: 1, diamond: 1, gator: 3 },
  { cherry: 9, bar: 1, seven: 4, diamond: 1, gator: 1 },
  { cherry: 1, bar: 8, seven: 4, diamond: 1, gator: 2 },
];

/** Multipliers of the bet. Cherries also pay on a pair; everything else needs three of a kind. */
export const SLOT_PAYTABLE = {
  twoCherries: 2,
  threeCherries: 5,
  threeBars: 10,
  threeSevens: 50,
  threeGators: 100,
  threeDiamonds: 500,
} as const;

export type SlotReels = [SlotSymbol, SlotSymbol, SlotSymbol];

export interface SlotResult {
  reels: SlotReels;
  bet: number;
  multiplier: number;
  /** Chips returned to the player (0 when nothing lined up). */
  payout: number;
  label: string;
  jackpot: boolean;
}

function totalWeight(reel: number): number {
  const w = REEL_WEIGHTS[reel];
  let sum = 0;
  for (const s of SLOT_SYMBOLS) sum += w[s];
  return sum;
}

export function isValidBet(bet: number): boolean {
  return SLOT_BETS.includes(bet);
}

export function isSlotSymbol(value: unknown): value is SlotSymbol {
  return typeof value === 'string' && (SLOT_SYMBOLS as readonly string[]).includes(value);
}

export function pickSymbol(rng: () => number, reel: number): SlotSymbol {
  const w = REEL_WEIGHTS[reel];
  let roll = rng() * totalWeight(reel);
  for (const s of SLOT_SYMBOLS) {
    roll -= w[s];
    if (roll < 0) return s;
  }
  return SLOT_SYMBOLS[SLOT_SYMBOLS.length - 1];
}

/** One full reel strip in weight proportions (for the spinning animation, not for odds). */
export function reelStrip(reel: number): SlotSymbol[] {
  const w = REEL_WEIGHTS[reel];
  const strip: SlotSymbol[] = [];
  const remaining: Record<SlotSymbol, number> = { ...w };
  const total = totalWeight(reel);
  // Smooth weighted round-robin keeps the strip from showing long runs of one symbol.
  for (let i = 0; i < total; i++) {
    let best: SlotSymbol | null = null;
    let bestRatio = -1;
    for (const s of SLOT_SYMBOLS) {
      if (remaining[s] <= 0) continue;
      const ratio = remaining[s] / w[s];
      if (ratio > bestRatio) {
        bestRatio = ratio;
        best = s;
      }
    }
    if (best === null) break;
    remaining[best]--;
    strip.push(best);
  }
  return strip;
}

export function evaluate(reels: SlotReels, bet: number): SlotResult {
  if (!isValidBet(bet)) throw new Error(`slots: bet ${bet} not in ${SLOT_BETS.join('/')}`);
  const [a, b, c] = reels;
  let multiplier = 0;
  let label = 'No win';
  let jackpot = false;
  if (a === b && b === c) {
    switch (a) {
      case 'diamond':
        multiplier = SLOT_PAYTABLE.threeDiamonds;
        label = 'JACKPOT! Three diamonds';
        jackpot = true;
        break;
      case 'gator':
        multiplier = SLOT_PAYTABLE.threeGators;
        label = 'Three gators';
        break;
      case 'seven':
        multiplier = SLOT_PAYTABLE.threeSevens;
        label = 'Three sevens';
        break;
      case 'bar':
        multiplier = SLOT_PAYTABLE.threeBars;
        label = 'Three bars';
        break;
      case 'cherry':
        multiplier = SLOT_PAYTABLE.threeCherries;
        label = 'Three cherries';
        break;
    }
  } else if (Number(a === 'cherry') + Number(b === 'cherry') + Number(c === 'cherry') === 2) {
    multiplier = SLOT_PAYTABLE.twoCherries;
    label = 'Two cherries';
  }
  return { reels, bet, multiplier, payout: multiplier * bet, label, jackpot };
}

export function spin(bet: number, rng: () => number): SlotResult {
  if (!isValidBet(bet)) throw new Error(`slots: bet ${bet} not in ${SLOT_BETS.join('/')}`);
  return evaluate([pickSymbol(rng, 0), pickSymbol(rng, 1), pickSymbol(rng, 2)], bet);
}

/** Exact expected return per chip wagered, for tests and the progress doc. */
export function expectedReturn(): number {
  const probs = REEL_WEIGHTS.map((w, reel) => {
    const total = totalWeight(reel);
    return SLOT_SYMBOLS.map((s) => w[s] / total);
  });
  let rtp = 0;
  for (let a = 0; a < 5; a++) {
    for (let b = 0; b < 5; b++) {
      for (let c = 0; c < 5; c++) {
        const p = probs[0][a] * probs[1][b] * probs[2][c];
        rtp += p * evaluate([SLOT_SYMBOLS[a], SLOT_SYMBOLS[b], SLOT_SYMBOLS[c]], 10).multiplier;
      }
    }
  }
  return rtp;
}
