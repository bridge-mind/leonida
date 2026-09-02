import { parseCard } from '../../casino/cards';
import { WHEEL_SEGMENT_COUNT } from '../../casino/LuckyWheel';
import { isValidPocket } from '../../casino/Roulette';
import { isSlotSymbol, type SlotReels } from '../../casino/Slots';

export type RigGame = 'blackjack' | 'roulette' | 'slots' | 'wheel';

/**
 * Deterministic outcomes for the next round of each game (debug action `casino.rig`).
 * Each value is consumed once by the panel that starts the round.
 */
export class CasinoRig {
  blackjack: string[] | null = null;
  roulette: number | null = null;
  slots: SlotReels | null = null;
  wheel: number | null = null;

  /** Accepts a JSON string or an already-parsed value; throws with a readable message on bad input. */
  set(game: string, raw: unknown): void {
    const value: unknown = typeof raw === 'string' ? parseLoose(raw) : raw;
    switch (game) {
      case 'blackjack': {
        if (!Array.isArray(value) || value.length < 4) throw new Error('rig blackjack expects ≥4 card codes, e.g. ["AS","9D","KH","7C"]');
        const codes = value.map((c) => String(c));
        for (const c of codes) parseCard(c);
        this.blackjack = codes;
        return;
      }
      case 'roulette': {
        const pocket = Number(value);
        if (!isValidPocket(pocket)) throw new Error('rig roulette expects a pocket 0..37 (37 = 00)');
        this.roulette = pocket;
        return;
      }
      case 'slots': {
        if (!Array.isArray(value) || value.length !== 3 || !value.every(isSlotSymbol)) {
          throw new Error('rig slots expects three symbols, e.g. ["diamond","diamond","diamond"]');
        }
        this.slots = [value[0], value[1], value[2]];
        return;
      }
      case 'wheel': {
        const index = Number(value);
        if (!Number.isInteger(index) || index < 0 || index >= WHEEL_SEGMENT_COUNT) throw new Error(`rig wheel expects a segment 0..${WHEEL_SEGMENT_COUNT - 1}`);
        this.wheel = index;
        return;
      }
      default:
        throw new Error(`rig: unknown game '${game}'`);
    }
  }

  takeBlackjack(): string[] | null {
    const v = this.blackjack;
    this.blackjack = null;
    return v;
  }

  takeRoulette(): number | null {
    const v = this.roulette;
    this.roulette = null;
    return v;
  }

  takeSlots(): SlotReels | null {
    const v = this.slots;
    this.slots = null;
    return v;
  }

  takeWheel(): number | null {
    const v = this.wheel;
    this.wheel = null;
    return v;
  }

  clear(): void {
    this.blackjack = null;
    this.roulette = null;
    this.slots = null;
    this.wheel = null;
  }
}

/** JSON first; otherwise a bare number or a comma/space separated list of tokens. */
function parseLoose(text: string): unknown {
  const trimmed = text.trim();
  try {
    return JSON.parse(trimmed) as unknown;
  } catch {
    if (/^-?\d+$/.test(trimmed)) return Number(trimmed);
    return trimmed.split(/[\s,;|]+/).filter(Boolean);
  }
}
