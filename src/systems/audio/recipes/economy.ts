import type { SoundId } from '../sounds';
import type { Synth } from '../synth';
import type { Recipe } from './types';

/** Rising/falling sine arpeggio used by cash and shop feedback. */
function arpRecipe(freqs: readonly number[], step: number, dur: number, vol: number, type: OscillatorType = 'sine', reverb = 0.12): Recipe {
  return { reverb, build: (s) => s.arp(s.t, freqs, step, dur, vol, type) };
}

function bell(s: Synth, at: number, freq: number, vol: number): void {
  s.ping(at, freq, 0.6, vol, 40);
  s.tone({ at, freq: freq * 2.76, dur: 0.35, vol: vol * 0.3, attack: 0.001 });
}

function crunch(s: Synth, at: number, vol: number): void {
  s.noiseBurst({ at, dur: 0.07, vol, filter: 'bandpass', freq: s.rand(1200, 2200), q: 1.2, attack: 0.002 });
  s.click(at + 0.01, vol * 0.6, s.rand(2500, 4000), 3, 0.01);
}

type EconomyId = Extract<
  SoundId,
  | `pickup.${string}`
  | `cash.${string}`
  | `register.${string}`
  | `robbery.${string}`
  | `store.${string}`
  | `shop.${string}`
  | `food.${string}`
  | `atm.${string}`
  | `respray.${string}`
  | `chopshop.${string}`
  | `gas.${string}`
>;

export const ECONOMY_RECIPES: Record<EconomyId, Recipe> = {
  'pickup.cash': arpRecipe([880, 1320], 0.06, 0.16, 0.3),
  'pickup.weapon': {
    build: (s) => {
      s.click(s.t, 0.4, 1600, 3);
      s.tone({ freq: 330, dur: 0.18, vol: 0.3, type: 'triangle' });
      s.ping(s.t + 0.04, 2400, 0.15, 0.2, 25);
    },
  },
  'pickup.ammo': {
    build: (s) => {
      s.click(s.t, 0.35, 1500, 3);
      s.click(s.t + 0.05, 0.35, 2000, 3);
      s.tone({ at: s.t + 0.05, freq: 660, dur: 0.12, vol: 0.25 });
    },
  },
  'pickup.armor': {
    build: (s) => {
      s.tone({ freq: 440, glideTo: 660, dur: 0.22, vol: 0.3, type: 'triangle', shape: 'hold', release: 0.08 });
      s.click(s.t, 0.3, 2200, 3);
    },
  },
  'pickup.health': arpRecipe([660, 990, 1320], 0.06, 0.2, 0.28, 'triangle'),
  'cash.gain': arpRecipe([784, 988, 1175], 0.07, 0.22, 0.3),
  'cash.loss': arpRecipe([660, 523, 415], 0.09, 0.25, 0.3),
  'cash.tick': {
    build: (s) => {
      s.click(s.t, 0.15, 3000, 4, 0.01);
      s.tone({ freq: 2000, dur: 0.03, vol: 0.12 });
    },
  },
  'register.open': {
    reverb: 0.2,
    build: (s) => {
      s.thump(s.t, 200, 120, 0.06, 0.4);
      s.click(s.t, 0.4, 800, 3, 0.02);
      s.noiseBurst({ at: s.t + 0.02, dur: 0.1, vol: 0.2, filter: 'lowpass', freq: 1500, attack: 0.02 });
      bell(s, s.t + 0.12, 2800, 0.35);
    },
  },
  'register.cash': {
    build: (s) => {
      s.noiseBurst({ kind: 'pink', dur: 0.15, vol: 0.3, filter: 'bandpass', freq: 2500, q: 1, attack: 0.01 });
      s.click(s.t + 0.1, 0.3, 1800, 3);
    },
  },
  'robbery.bagfill.loop': {
    build: (s) => {
      const g = s.gain(0.15);
      const bp = s.filter('bandpass', 2000, 1, g);
      s.noiseSrc('pink', bp);
      s.periodic(
        () => s.rand(0.1, 0.28),
        (at) => s.noiseBurst({ kind: 'pink', at, dur: s.rand(0.05, 0.12), vol: s.rand(0.15, 0.3), filter: 'bandpass', freq: s.rand(1500, 3500), q: 1.2, attack: 0.01 }),
      );
    },
  },
  'robbery.complete': {
    reverb: 0.2,
    build: (s) => {
      s.arp(s.t, [523, 659, 784, 1047], 0.07, 0.3, 0.3, 'triangle');
      s.noiseBurst({ dur: 0.2, vol: 0.15, filter: 'bandpass', freq: 1200, freqTo: 4000, q: 1 });
    },
  },
  'store.bell': {
    reverb: 0.3,
    build: (s) => {
      bell(s, s.t, 2400, 0.35);
      bell(s, s.t + 0.02, 3600, 0.15);
    },
  },
  'store.closed': {
    build: (s) => {
      const lp = s.filter('lowpass', 900, 1);
      s.tone({ freq: 110, dur: 0.3, vol: 0.3, type: 'square', shape: 'hold', dest: lp });
      s.tone({ freq: 400, glideTo: 300, dur: 0.3, vol: 0.2, type: 'triangle' });
    },
  },
  'shop.buy': {
    build: (s) => {
      s.arp(s.t, [880, 1175], 0.07, 0.2, 0.3);
      s.click(s.t + 0.14, 0.2, 3000, 4, 0.01);
    },
  },
  'shop.sell': arpRecipe([1175, 880], 0.07, 0.2, 0.3),
  'shop.denied': {
    build: (s) => {
      const lp = s.filter('lowpass', 700, 1);
      s.tone({ freq: 150, dur: 0.14, vol: 0.3, type: 'square', shape: 'hold', release: 0.02, dest: lp });
      s.tone({ at: s.t + 0.18, freq: 150, dur: 0.14, vol: 0.3, type: 'square', shape: 'hold', release: 0.02, dest: lp });
    },
  },
  'food.eat': {
    build: (s) => {
      crunch(s, s.t, 0.35);
      crunch(s, s.t + 0.16, 0.3);
      crunch(s, s.t + 0.3, 0.3);
      s.thump(s.t + 0.5, 200, 100, 0.12, 0.25);
    },
  },
  'atm.beep': {
    build: (s) => {
      s.tone({ freq: 1400, dur: 0.09, vol: 0.25, shape: 'hold', release: 0.01 });
      s.tone({ at: s.t + 0.14, freq: 1400, dur: 0.09, vol: 0.25, shape: 'hold', release: 0.01 });
    },
  },
  'respray.spray.loop': {
    build: (s) => {
      const g = s.gain(0.3);
      const bp = s.filter('bandpass', 4000, 0.7, g);
      s.noiseSrc('white', bp);
      s.lfo(0.7, 0.2, g.gain, 'sine');
      s.lfo(0.7, 800, bp.frequency, 'triangle');
    },
  },
  'respray.done': {
    reverb: 0.15,
    build: (s) => {
      s.arp(s.t, [659, 880], 0.08, 0.25, 0.3, 'triangle');
      s.noiseBurst({ dur: 0.3, vol: 0.15, filter: 'bandpass', freq: 4000, q: 0.8, attack: 0.02 });
    },
  },
  'chopshop.crush': {
    reverb: 0.25,
    build: (s) => {
      s.noiseBurst({ dur: 1.2, vol: 0.6, filter: 'lowpass', freq: 1500, freqTo: 200, q: 0.8, attack: 0.02, sweepTime: 1.2 });
      s.thump(s.t, 50, 30, 0.6, 0.6);
      for (let i = 0; i < 7; i++) s.ping(s.t + s.rand(0.05, 0.9), s.rand(800, 1700), 0.25, 0.3, 12);
    },
  },
  'chopshop.paid': {
    reverb: 0.15,
    build: (s) => {
      s.arp(s.t, [784, 988, 1175, 1568], 0.07, 0.25, 0.3);
      bell(s, s.t + 0.3, 2600, 0.25);
    },
  },
  'gas.pump.loop': {
    build: (s) => {
      const g = s.gain(0.3);
      const lp = s.filter('lowpass', 400, 1, g);
      s.osc('sawtooth', 55, lp);
      const gurgle = s.gain(0.25, g);
      const gurgleBp = s.filter('bandpass', 420, 1.2, gurgle);
      s.noiseSrc('pink', gurgleBp);
      s.lfo(4, 0.15, gurgle.gain);
      s.periodic(
        () => 1,
        (at) => s.click(at, 0.2, 2200, 4, 0.01),
      );
    },
  },
};
