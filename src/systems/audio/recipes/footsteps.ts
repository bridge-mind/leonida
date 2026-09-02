import type { NoiseKind } from '../noise';
import type { SoundId } from '../sounds';
import type { Synth } from '../synth';
import { voiceLine } from './peds';
import type { Recipe } from './types';

interface Step {
  kind?: NoiseKind;
  filter: BiquadFilterType;
  freq: number;
  q: number;
  dur: number;
  vol: number;
  thumpHz?: number;
  ring?: number;
}

/** Filtered noise tick; the driver adds ±10 % pitch and the recipe jitters the filter. */
function step(p: Step): Recipe {
  return {
    build: (s) => {
      s.noiseBurst({ kind: p.kind, dur: p.dur, vol: p.vol, filter: p.filter, freq: s.jitter(p.freq, 0.1), q: p.q, attack: 0.002, release: 0.02 });
      if (p.thumpHz) s.thump(s.t, s.jitter(p.thumpHz, 0.1), p.thumpHz * 0.6, p.dur * 1.2, p.vol * 0.5);
      if (p.ring) s.ping(s.t, s.jitter(p.ring, 0.1), 0.12, p.vol * 0.4, 20);
    },
  };
}

export function splash(s: Synth, big: boolean): void {
  const k = big ? 1.8 : 1;
  s.noiseBurst({ dur: 0.28 * k, vol: 0.5 * Math.min(k, 1.4), filter: 'lowpass', freq: 400, freqTo: 3500, q: 0.8, attack: 0.01 });
  s.noiseBurst({ kind: 'brown', at: s.t + 0.05, dur: 0.5 * k, vol: 0.35, filter: 'lowpass', freq: 900, freqTo: 250, attack: 0.03 });
  s.thump(s.t, 120, 60, 0.2 * k, 0.4);
  if (big) for (let i = 0; i < 6; i++) s.noiseBurst({ at: s.t + 0.2 + i * 0.09, dur: 0.06, vol: 0.15, filter: 'bandpass', freq: s.rand(1500, 4000), q: 2 });
}

type FootId = Extract<SoundId, `footstep.${string}` | `player.${string}` | `water.${string}`>;

export const FOOTSTEP_RECIPES: Record<FootId, Recipe> = {
  'footstep.concrete': step({ filter: 'bandpass', freq: 2200, q: 1, dur: 0.05, vol: 0.5, thumpHz: 140 }),
  'footstep.asphalt': step({ filter: 'bandpass', freq: 1700, q: 1, dur: 0.06, vol: 0.45, thumpHz: 130 }),
  'footstep.grass': step({ kind: 'pink', filter: 'lowpass', freq: 900, q: 0.8, dur: 0.09, vol: 0.4 }),
  'footstep.sand': step({ kind: 'pink', filter: 'lowpass', freq: 1200, q: 0.7, dur: 0.1, vol: 0.4 }),
  'footstep.dirt': step({ kind: 'pink', filter: 'lowpass', freq: 700, q: 0.8, dur: 0.09, vol: 0.45, thumpHz: 100 }),
  'footstep.wood': step({ filter: 'bandpass', freq: 600, q: 2.5, dur: 0.08, vol: 0.5, thumpHz: 170 }),
  'footstep.metal': step({ filter: 'bandpass', freq: 1400, q: 1.5, dur: 0.06, vol: 0.45, thumpHz: 120, ring: 1900 }),
  'footstep.water': {
    build: (s) => {
      s.noiseBurst({ dur: 0.14, vol: 0.4, filter: 'lowpass', freq: 600, freqTo: 3000, q: 0.8, attack: 0.005 });
      s.thump(s.t, 110, 70, 0.1, 0.2);
    },
  },
  'player.jump': {
    build: (s) => {
      s.whoosh(s.t, 300, 1200, 0.2, 0.25);
      s.noiseBurst({ dur: 0.05, vol: 0.2, filter: 'bandpass', freq: 1800, q: 1 });
    },
  },
  'player.land': {
    build: (s) => {
      s.thump(s.t, 95, 55, 0.14, 0.45);
      s.noiseBurst({ dur: 0.07, vol: 0.35, filter: 'lowpass', freq: 700, attack: 0.002 });
    },
  },
  'player.land.hard': {
    build: (s) => {
      s.thump(s.t, 70, 35, 0.3, 0.7);
      s.noiseBurst({ kind: 'brown', dur: 0.2, vol: 0.5, filter: 'lowpass', freq: 500, attack: 0.002 });
      voiceLine(s, s.t + 0.02, { f0: 150, f1: 110, f2: 90, dur: 0.2, formant: 500, vol: 0.3 });
    },
  },
  'player.hurt': { build: (s) => voiceLine(s, s.t, { f0: 180, f1: 240, f2: 140, dur: 0.22, formant: 700, vol: 0.4 }) },
  'player.death': {
    reverb: 0.3,
    build: (s) => {
      voiceLine(s, s.t, { f0: 200, f1: 140, f2: 70, dur: 0.7, formant: 600, vol: 0.45 });
      s.thump(s.t + 0.5, 70, 40, 0.25, 0.5);
      s.noiseBurst({ kind: 'brown', at: s.t + 0.5, dur: 0.2, vol: 0.4, filter: 'lowpass', freq: 500 });
    },
  },
  'player.heartbeat.loop': {
    build: (s) => {
      const lp = s.filter('lowpass', 180, 0.8);
      s.periodic(
        () => 0.9,
        (at) => {
          s.tone({ at, freq: 55, glideTo: 40, dur: 0.14, vol: 0.9, attack: 0.008, dest: lp });
          s.tone({ at: at + 0.18, freq: 50, glideTo: 36, dur: 0.12, vol: 0.6, attack: 0.008, dest: lp });
        },
      );
    },
  },
  'player.swim.loop': {
    build: (s) => {
      const g = s.gain(0.35);
      const lp = s.filter('lowpass', 900, 0.7, g);
      s.noiseSrc('pink', lp);
      s.lfo(1.2, 0.25, g.gain);
      s.periodic(
        () => s.rand(0.7, 1.1),
        (at) => s.noiseBurst({ at, dur: 0.18, vol: 0.25, filter: 'lowpass', freq: 500, freqTo: 2500, attack: 0.02 }),
      );
    },
  },
  'water.splash': { build: (s) => splash(s, false) },
  'water.splash.big': { reverb: 0.2, build: (s) => splash(s, true) },
};
