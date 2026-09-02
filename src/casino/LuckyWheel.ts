/**
 * The Lucky Wheel: 20 weighted segments, one spin per game day.
 * Pure logic; the owning system applies prizes (cash, chips, mystery item, vehicle).
 */
export type WheelPrize =
  | { kind: 'cash'; amount: number }
  | { kind: 'chips'; amount: number }
  | { kind: 'mystery' }
  | { kind: 'vehicle' };

export interface WheelSegment {
  prize: WheelPrize;
  /** Relative probability; the pointer visually lands on the chosen segment. */
  weight: number;
  label: string;
  color: string;
}

const cash = (amount: number, weight: number, color: string): WheelSegment => ({
  prize: { kind: 'cash', amount },
  weight,
  label: `$${amount.toLocaleString('en-US')}`,
  color,
});
const chips = (amount: number, weight: number): WheelSegment => ({
  prize: { kind: 'chips', amount },
  weight,
  label: `${amount.toLocaleString('en-US')} CHIPS`,
  color: '#35e6ff',
});
const mystery: WheelSegment = { prize: { kind: 'mystery' }, weight: 5, label: 'MYSTERY', color: '#a35bff' };
const vehicle: WheelSegment = { prize: { kind: 'vehicle' }, weight: 1, label: 'SPORTS CAR', color: '#ff4fa3' };

export const WHEEL_SEGMENTS: readonly WheelSegment[] = [
  cash(500, 10, '#4ee06a'),
  chips(1000, 8),
  cash(1000, 8, '#3fbf5a'),
  mystery,
  cash(500, 10, '#4ee06a'),
  cash(2500, 5, '#ffd23f'),
  chips(2500, 4),
  cash(750, 9, '#4ee06a'),
  mystery,
  cash(500, 10, '#4ee06a'),
  vehicle,
  cash(1000, 8, '#3fbf5a'),
  chips(1000, 8),
  cash(5000, 2, '#ffb347'),
  cash(500, 10, '#4ee06a'),
  mystery,
  cash(1000, 8, '#3fbf5a'),
  chips(2500, 4),
  cash(10000, 1, '#ff8a1f'),
  cash(750, 9, '#4ee06a'),
];

export const WHEEL_SEGMENT_COUNT = WHEEL_SEGMENTS.length;

export type MysteryPrize =
  | { kind: 'weapon'; weaponId: 'smg'; ammo: number }
  | { kind: 'ammo'; weaponId: 'pistol'; ammo: number }
  | { kind: 'armor'; amount: number };

export function rollMystery(rng: () => number): MysteryPrize {
  const r = rng();
  if (r < 0.4) return { kind: 'weapon', weaponId: 'smg', ammo: 60 };
  if (r < 0.7) return { kind: 'ammo', weaponId: 'pistol', ammo: 100 };
  return { kind: 'armor', amount: 100 };
}

export interface WheelSpin {
  index: number;
  segment: WheelSegment;
}

export class LuckyWheel {
  /** Game day of the last spin; 0 = never. */
  lastSpinDay = 0;
  private readonly totalWeight: number;

  constructor(private readonly rng: () => number) {
    let sum = 0;
    for (const s of WHEEL_SEGMENTS) sum += s.weight;
    this.totalWeight = sum;
  }

  canSpin(day: number): boolean {
    return day !== this.lastSpinDay;
  }

  /** Weighted segment pick; exposed so the UI can preview odds and tests can check the distribution. */
  pickIndex(): number {
    let roll = this.rng() * this.totalWeight;
    for (let i = 0; i < WHEEL_SEGMENTS.length; i++) {
      roll -= WHEEL_SEGMENTS[i].weight;
      if (roll < 0) return i;
    }
    return WHEEL_SEGMENTS.length - 1;
  }

  /** Consumes today's spin. Returns null when already spun today. `forcedIndex` is for rigs/tests. */
  spin(day: number, forcedIndex?: number): WheelSpin | null {
    if (!this.canSpin(day)) return null;
    const index =
      forcedIndex !== undefined && Number.isInteger(forcedIndex) && forcedIndex >= 0 && forcedIndex < WHEEL_SEGMENTS.length
        ? forcedIndex
        : this.pickIndex();
    this.lastSpinDay = day;
    return { index, segment: WHEEL_SEGMENTS[index] };
  }
}

/** Angle (radians) of a segment centre, measured clockwise from the pointer at the top. */
export function segmentAngle(index: number): number {
  return ((index + 0.5) / WHEEL_SEGMENT_COUNT) * Math.PI * 2;
}
