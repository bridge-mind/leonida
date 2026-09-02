import type { SoundId } from '../sounds';
import type { Synth } from '../synth';

export interface Recipe {
  /** Base gain multiplied into the voice volume (default 1). */
  gain?: number;
  /** Reverb send amount for the whole voice (0 = dry). */
  reverb?: number;
  /** Builds the sound at `s.t`. One-shots must leave `s.endTime` set (helpers do it). */
  build(s: Synth): void;
}

/** Exhaustive recipe table for the ids matching a prefix, checked at compile time. */
export type RecipeTable<P extends string> = { readonly [K in Extract<SoundId, `${P}${string}`>]: Recipe };

export type Ids<P extends string> = Extract<SoundId, `${P}${string}`>;
