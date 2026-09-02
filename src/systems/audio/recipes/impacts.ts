import type { NoiseKind } from '../noise';
import type { Recipe } from './types';
import type { SoundId } from '../sounds';

interface Impact {
  kind?: NoiseKind;
  filter: BiquadFilterType;
  freq: number;
  freqTo?: number;
  q?: number;
  dur: number;
  vol: number;
  thumpHz?: number;
}

/** Short filtered noise, optionally with a low thump underneath. */
function impact(i: Impact): Recipe {
  return {
    build: (s) => {
      s.noiseBurst({ kind: i.kind, dur: i.dur, vol: i.vol, filter: i.filter, freq: s.jitter(i.freq, 0.08), freqTo: i.freqTo, q: i.q ?? 1, attack: 0.001 });
      if (i.thumpHz) s.thump(s.t, i.thumpHz, i.thumpHz * 0.6, i.dur, i.vol * 0.6);
    },
  };
}

type ImpactId = Extract<SoundId, `impact.${string}`> | 'bullet.whizz' | 'glass.shatter';

export const IMPACT_RECIPES: Record<ImpactId, Recipe> = {
  'impact.concrete': impact({ filter: 'bandpass', freq: 2600, q: 0.7, dur: 0.08, vol: 0.55, thumpHz: 180 }),
  'impact.asphalt': impact({ filter: 'bandpass', freq: 1900, q: 0.8, dur: 0.09, vol: 0.5, thumpHz: 150 }),
  'impact.metal': {
    build: (s) => {
      s.ping(s.t, s.jitter(2200, 0.15), 0.28, 0.5, 22);
      s.ping(s.t, s.jitter(1500, 0.1), 0.2, 0.3, 18);
      s.click(s.t, 0.4, 3000, 2, 0.012);
    },
  },
  'impact.glass': {
    build: (s) => {
      for (let i = 0; i < 4; i++) s.ping(s.t + i * 0.025, s.rand(3200, 7500), 0.12, 0.28, 35);
      s.noiseBurst({ dur: 0.05, vol: 0.3, filter: 'highpass', freq: 4000, attack: 0.001 });
    },
  },
  'impact.wood': impact({ filter: 'bandpass', freq: 700, q: 1.5, dur: 0.1, vol: 0.5, thumpHz: 220 }),
  'impact.dirt': impact({ kind: 'pink', filter: 'lowpass', freq: 900, dur: 0.12, vol: 0.45, thumpHz: 120 }),
  'impact.sand': impact({ kind: 'pink', filter: 'lowpass', freq: 1200, dur: 0.1, vol: 0.4 }),
  'impact.water': impact({ filter: 'lowpass', freq: 500, freqTo: 3500, dur: 0.22, vol: 0.5, thumpHz: 140 }),
  'impact.flesh': impact({ kind: 'pink', filter: 'lowpass', freq: 700, dur: 0.1, vol: 0.45, thumpHz: 110 }),
  'impact.vehicle': {
    build: (s) => {
      s.ping(s.t, s.jitter(1100, 0.1), 0.18, 0.4, 12);
      s.thump(s.t, 160, 80, 0.12, 0.45);
      s.noiseBurst({ dur: 0.06, vol: 0.4, filter: 'bandpass', freq: 2400, q: 0.8, attack: 0.001 });
    },
  },
  'impact.ricochet': {
    build: (s) => {
      s.click(s.t, 0.4, 3000, 2, 0.01);
      s.tone({ freq: s.rand(2600, 3600), glideTo: 700, dur: 0.32, vol: 0.3, attack: 0.002 });
    },
  },
  'bullet.whizz': {
    build: (s) => {
      s.noiseBurst({ dur: 0.14, vol: 0.4, filter: 'bandpass', freq: 3800, freqTo: 600, q: 4, attack: 0.005, rate: 1.4 });
    },
  },
  'glass.shatter': {
    reverb: 0.2,
    build: (s) => {
      s.noiseBurst({ dur: 0.12, vol: 0.5, filter: 'highpass', freq: 3000, attack: 0.001 });
      for (let i = 0; i < 12; i++) s.ping(s.t + s.rand(0, 0.45), s.rand(2800, 8000), s.rand(0.08, 0.2), 0.25, 30);
    },
  },
};
