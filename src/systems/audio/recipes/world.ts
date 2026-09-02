import type { SoundId } from '../sounds';
import type { Recipe } from './types';

type WorldId = Extract<
  SoundId,
  `door.${string}` | `interior.${string}` | `garage.${string}` | `lamp.${string}` | `neon.${string}` | `traffic.${string}` | `save.${string}`
>;

export const WORLD_RECIPES: Record<WorldId, Recipe> = {
  'door.open': {
    build: (s) => {
      s.click(s.t, 0.4, 500, 3, 0.02);
      const bp = s.filter('bandpass', 700, 4);
      s.tone({ at: s.t + 0.03, freq: 300, glideTo: 380, dur: 0.3, vol: 0.12, type: 'sawtooth', shape: 'hold', attack: 0.05, release: 0.08, dest: bp });
    },
  },
  'door.close': {
    build: (s) => {
      s.thump(s.t, 130, 70, 0.1, 0.5);
      s.click(s.t + 0.02, 0.4, 900, 3, 0.02);
      s.noiseBurst({ dur: 0.06, vol: 0.25, filter: 'lowpass', freq: 900 });
    },
  },
  'door.locked': {
    build: (s) => {
      s.click(s.t, 0.4, 600, 4, 0.02);
      s.click(s.t + 0.12, 0.35, 650, 4, 0.02);
      s.noiseBurst({ at: s.t + 0.14, dur: 0.08, vol: 0.15, filter: 'bandpass', freq: 1800, q: 2 });
    },
  },
  'interior.enter': {
    build: (s) => {
      s.noiseBurst({ dur: 0.3, vol: 0.2, filter: 'bandpass', freq: 400, freqTo: 1200, q: 1, attack: 0.1 });
      s.click(s.t + 0.3, 0.3, 1200, 3, 0.015);
    },
  },
  'interior.exit': {
    build: (s) => {
      s.noiseBurst({ dur: 0.3, vol: 0.2, filter: 'bandpass', freq: 1200, freqTo: 400, q: 1, attack: 0.1 });
      s.click(s.t + 0.3, 0.3, 900, 3, 0.015);
    },
  },
  'garage.door.loop': {
    build: (s) => {
      const g = s.gain(0.35);
      const lp = s.filter('lowpass', 300, 1, g);
      s.osc('sawtooth', 45, lp);
      s.osc('square', 90, lp);
      s.periodic(
        () => 0.08,
        (at) => s.click(at, s.rand(0.08, 0.2), s.rand(1500, 3000), 3, 0.01),
      );
    },
  },
  'lamp.buzz.loop': {
    build: (s) => {
      const g = s.gain(0.12);
      const lp = s.filter('lowpass', 1200, 1, g);
      s.osc('sawtooth', 120, lp);
      s.osc('sawtooth', 240, lp);
      s.periodic(
        () => s.rand(0.3, 1.5),
        (at) => s.click(at, 0.1, s.rand(2000, 4000), 3, 0.008),
      );
    },
  },
  'neon.buzz.loop': {
    build: (s) => {
      const g = s.gain(0.1);
      const bp = s.filter('bandpass', 2400, 3, g);
      s.osc('sawtooth', 100, bp);
      s.osc('sawtooth', 120, bp);
      s.lfo(0.4, 0.03, g.gain);
    },
  },
  'traffic.light.tick': { build: (s) => s.click(s.t, 0.15, 2000, 4, 0.01) },
  'save.complete': { reverb: 0.25, build: (s) => s.arp(s.t, [659, 880, 1108], 0.08, 0.3, 0.25, 'triangle') },
};
