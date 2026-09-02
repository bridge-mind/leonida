/**
 * American roulette (0, 00, 1–36). Pure logic: bet validation on the standard
 * layout grid, settlement and the wheel order used by the 3D wheel animation.
 *
 * Pockets are integers 0..37 where 37 is `00`. Chips flow: the caller deducts a
 * bet when it is placed and credits `settle().perBet[i]` (bet + winnings) afterwards.
 */
export const DOUBLE_ZERO = 37;
export const POCKET_COUNT = 38;

export type PocketColor = 'red' | 'black' | 'green';

const RED_NUMBERS: ReadonlySet<number> = new Set([1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]);

/** Clockwise pocket order on a real American wheel, starting at 0. */
export const WHEEL_ORDER: readonly number[] = [
  0, 28, 9, 26, 30, 11, 7, 20, 32, 17, 5, 22, 34, 15, 3, 24, 36, 13, 1, DOUBLE_ZERO, 27, 10, 25, 29, 12, 8, 19, 31, 18, 6,
  21, 33, 16, 4, 23, 35, 14, 2,
];

export function pocketLabel(pocket: number): string {
  return pocket === DOUBLE_ZERO ? '00' : String(pocket);
}

export function pocketColor(pocket: number): PocketColor {
  if (pocket === 0 || pocket === DOUBLE_ZERO) return 'green';
  return RED_NUMBERS.has(pocket) ? 'red' : 'black';
}

export function isValidPocket(pocket: number): boolean {
  return Number.isInteger(pocket) && pocket >= 0 && pocket < POCKET_COUNT;
}

export type InsideBetKind = 'straight' | 'split' | 'street' | 'corner';
export type EvenMoneyKind = 'red' | 'black' | 'odd' | 'even' | 'low' | 'high';
export type GroupBetKind = 'dozen' | 'column';
export type RouletteBetKind = InsideBetKind | EvenMoneyKind | GroupBetKind;

export type RouletteBet =
  | { kind: InsideBetKind; pockets: number[]; amount: number }
  | { kind: EvenMoneyKind; amount: number }
  | { kind: GroupBetKind; index: 0 | 1 | 2; amount: number };

/** Winnings per chip wagered (the stake is returned on top). */
export const ROULETTE_ODDS: Record<RouletteBetKind, number> = {
  straight: 35,
  split: 17,
  street: 11,
  corner: 8,
  red: 1,
  black: 1,
  odd: 1,
  even: 1,
  low: 1,
  high: 1,
  dozen: 2,
  column: 2,
};

const INSIDE_SIZE: Record<InsideBetKind, number> = { straight: 1, split: 2, street: 3, corner: 4 };

/** Layout column (0..11) and row (0..2) of a number 1..36. */
function gridPos(n: number): { col: number; row: number } {
  return { col: Math.floor((n - 1) / 3), row: (n - 1) % 3 };
}

function isZero(p: number): boolean {
  return p === 0 || p === DOUBLE_ZERO;
}

function validSplit(a: number, b: number): boolean {
  if (isZero(a) || isZero(b)) {
    if (isZero(a) && isZero(b)) return a !== b;
    const zero = isZero(a) ? a : b;
    const n = isZero(a) ? b : a;
    return zero === 0 ? n === 1 || n === 2 : n === 2 || n === 3;
  }
  const pa = gridPos(a);
  const pb = gridPos(b);
  return (pa.col === pb.col && Math.abs(pa.row - pb.row) === 1) || (pa.row === pb.row && Math.abs(pa.col - pb.col) === 1);
}

export function isValidBet(bet: RouletteBet): boolean {
  if (!Number.isInteger(bet.amount) || bet.amount <= 0) return false;
  switch (bet.kind) {
    case 'straight':
    case 'split':
    case 'street':
    case 'corner': {
      const p = bet.pockets;
      if (p.length !== INSIDE_SIZE[bet.kind] || !p.every(isValidPocket) || new Set(p).size !== p.length) return false;
      if (bet.kind === 'straight') return true;
      if (bet.kind === 'split') return validSplit(p[0], p[1]);
      const sorted = [...p].sort((x, y) => x - y);
      if (sorted.some(isZero)) return false;
      if (bet.kind === 'street') return sorted[0] % 3 === 1 && sorted[1] === sorted[0] + 1 && sorted[2] === sorted[0] + 2;
      const g = gridPos(sorted[0]);
      return g.row < 2 && g.col < 11 && sorted[1] === sorted[0] + 1 && sorted[2] === sorted[0] + 3 && sorted[3] === sorted[0] + 4;
    }
    case 'dozen':
    case 'column':
      return bet.index === 0 || bet.index === 1 || bet.index === 2;
    default:
      return true;
  }
}

export function betCovers(bet: RouletteBet, outcome: number): boolean {
  switch (bet.kind) {
    case 'straight':
    case 'split':
    case 'street':
    case 'corner':
      return bet.pockets.includes(outcome);
    case 'red':
      return pocketColor(outcome) === 'red';
    case 'black':
      return pocketColor(outcome) === 'black';
    case 'odd':
      return !isZero(outcome) && outcome % 2 === 1;
    case 'even':
      return !isZero(outcome) && outcome % 2 === 0;
    case 'low':
      return outcome >= 1 && outcome <= 18;
    case 'high':
      return outcome >= 19 && outcome <= 36;
    case 'dozen':
      return !isZero(outcome) && Math.floor((outcome - 1) / 12) === bet.index;
    case 'column':
      return !isZero(outcome) && (outcome - 1) % 3 === bet.index;
  }
}

export interface RouletteSettlement {
  /** Chips returned per bet (stake + winnings, or 0). */
  perBet: number[];
  returned: number;
  wagered: number;
  net: number;
}

export function settle(bets: readonly RouletteBet[], outcome: number): RouletteSettlement {
  if (!isValidPocket(outcome)) throw new Error(`roulette: bad outcome ${outcome}`);
  const perBet: number[] = [];
  let returned = 0;
  let wagered = 0;
  for (const bet of bets) {
    wagered += bet.amount;
    const win = betCovers(bet, outcome) ? bet.amount * (ROULETTE_ODDS[bet.kind] + 1) : 0;
    perBet.push(win);
    returned += win;
  }
  return { perBet, returned, wagered, net: returned - wagered };
}

export function spin(rng: () => number): number {
  return Math.min(POCKET_COUNT - 1, Math.floor(rng() * POCKET_COUNT));
}

/** Angle (radians, clockwise from the 0 pocket) at the centre of a pocket on the wheel. */
export function pocketAngle(pocket: number): number {
  const idx = WHEEL_ORDER.indexOf(pocket);
  return ((idx < 0 ? 0 : idx) / WHEEL_ORDER.length) * Math.PI * 2;
}

export function describeBet(bet: RouletteBet): string {
  switch (bet.kind) {
    case 'straight':
      return `Straight ${pocketLabel(bet.pockets[0])}`;
    case 'split':
    case 'street':
    case 'corner':
      return `${bet.kind[0].toUpperCase()}${bet.kind.slice(1)} ${bet.pockets.map(pocketLabel).join('-')}`;
    case 'dozen':
      return `Dozen ${bet.index * 12 + 1}-${bet.index * 12 + 12}`;
    case 'column':
      return `Column ${bet.index + 1}`;
    case 'low':
      return '1-18';
    case 'high':
      return '19-36';
    default:
      return bet.kind[0].toUpperCase() + bet.kind.slice(1);
  }
}
