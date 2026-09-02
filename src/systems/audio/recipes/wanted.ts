import type { SoundId } from '../sounds';
import type { Synth } from '../synth';
import { radioSquelch } from './peds';
import type { Recipe } from './types';

/** Dark descending stinger: detuned saw pair gliding down an octave over a lowpass plus a sub boom. */
function darkStinger(freqs: readonly number[], dur: number, vol: number): Recipe {
  return {
    reverb: 0.45,
    build: (s) => {
      const lp = s.filter('lowpass', 800, 1);
      for (const f of freqs) {
        s.tone({ freq: f, glideTo: f * 0.5, glideTime: dur, dur, vol, type: 'sawtooth', shape: 'hold', attack: 0.02, release: 0.6, detune: -6, dest: lp });
        s.tone({ freq: f, glideTo: f * 0.5, glideTime: dur, dur, vol, type: 'sawtooth', shape: 'hold', attack: 0.02, release: 0.6, detune: 6, dest: lp });
      }
      s.tone({ freq: 50, glideTo: 30, dur: dur * 0.8, vol: 0.5, attack: 0.01 });
      s.noiseBurst({ kind: 'brown', dur: dur * 0.7, vol: 0.25, filter: 'lowpass', freq: 400, freqTo: 100, attack: 0.2 });
    },
  };
}

function pad(s: Synth, at: number, freqs: readonly number[], dur: number, vol: number, cutoff: number): void {
  const lp = s.filter('lowpass', cutoff, 0.8);
  for (const f of freqs) {
    s.tone({ at, freq: f, dur, vol, type: 'sawtooth', shape: 'hold', attack: 0.05, release: 0.5, detune: -5, dest: lp });
    s.tone({ at, freq: f, dur, vol, type: 'sawtooth', shape: 'hold', attack: 0.05, release: 0.5, detune: 5, dest: lp });
  }
}

function lowPulse(s: Synth, at: number, vol: number): void {
  const lp = s.filter('lowpass', 300, 1);
  s.tone({ at, freq: 110, dur: 0.4, vol, shape: 'hold', attack: 0.02, release: 0.15, dest: lp });
  s.tone({ at, freq: 55, dur: 0.4, vol: vol * 0.8, shape: 'hold', attack: 0.02, release: 0.15, dest: lp });
}

type WantedId = Extract<
  SoundId,
  | `wanted.${string}`
  | `police.${string}`
  | `mission.${string}`
  | `cutscene.${string}`
  | `dialogue.${string}`
  | `ability.${string}`
  | `character.${string}`
>;

export const WANTED_RECIPES: Record<WantedId, Recipe> = {
  'wanted.star.gain': {
    build: (s) => {
      s.tone({ freq: 660, glideTo: 990, dur: 0.16, vol: 0.3, attack: 0.005 });
      s.click(s.t, 0.25, 3000, 4, 0.01);
    },
  },
  'wanted.star.flash': {
    build: (s) => {
      s.tone({ freq: 1500, dur: 0.06, vol: 0.18, attack: 0.003 });
      s.click(s.t, 0.12, 4000, 4, 0.008);
    },
  },
  'wanted.lost': { reverb: 0.35, build: (s) => s.arp(s.t, [659, 784, 988], 0.13, 0.35, 0.3, 'triangle') },
  'wanted.searching': {
    build: (s) => {
      lowPulse(s, s.t, 0.35);
      lowPulse(s, s.t + 0.5, 0.2);
    },
  },
  'wanted.busted': darkStinger([220, 261.6], 1.2, 0.16),
  'wanted.wasted': darkStinger([196, 233.1], 1.6, 0.16),
  'wanted.hotscene': {
    build: (s) => {
      lowPulse(s, s.t, 0.3);
      s.tone({ freq: 440, dur: 0.4, vol: 0.15, type: 'triangle', shape: 'hold' });
      s.tone({ freq: 466, dur: 0.4, vol: 0.15, type: 'triangle', shape: 'hold' });
    },
  },
  'police.spotlight': {
    build: (s) => {
      s.click(s.t, 0.4, 1200, 3, 0.02);
      const lp = s.filter('lowpass', 400, 1);
      s.tone({ at: s.t + 0.02, freq: 120, dur: 0.6, vol: 0.25, type: 'sawtooth', shape: 'hold', attack: 0.05, release: 0.2, dest: lp });
    },
  },
  'police.spikes.deploy': {
    build: (s) => {
      for (let i = 0; i < 6; i++) s.click(s.t + i * 0.04, 0.35, 3000, 6, 0.01);
      s.ping(s.t + 0.25, 2200, 0.25, 0.35, 20);
    },
  },
  'police.roadblock': {
    build: (s) => {
      s.thump(s.t, 90, 50, 0.15, 0.5);
      s.noiseBurst({ dur: 0.08, vol: 0.3, filter: 'lowpass', freq: 800 });
      radioSquelch(s, s.t + 0.2, 0.25, 0.25);
    },
  },
  'police.dispatch': { build: (s) => radioSquelch(s, s.t, 0.3, 0.3) },
  'mission.start': {
    reverb: 0.3,
    build: (s) => {
      s.tone({ freq: 523, dur: 0.3, vol: 0.25, shape: 'hold', attack: 0.01, release: 0.15 });
      s.tone({ at: s.t + 0.18, freq: 784, dur: 0.45, vol: 0.25, shape: 'hold', attack: 0.01, release: 0.25 });
    },
  },
  'mission.objective': {
    reverb: 0.2,
    build: (s) => {
      s.tone({ freq: 880, dur: 0.3, vol: 0.22, type: 'triangle', attack: 0.005 });
      s.tone({ at: s.t + 0.1, freq: 1108, dur: 0.3, vol: 0.12, type: 'triangle', attack: 0.005 });
    },
  },
  'mission.checkpoint': { reverb: 0.2, build: (s) => s.arp(s.t, [784, 1047], 0.1, 0.3, 0.25, 'triangle') },
  'mission.passed': {
    reverb: 0.5,
    build: (s) => {
      s.arp(s.t, [523, 659, 784], 0.15, 0.6, 0.28, 'triangle');
      pad(s, s.t + 0.3, [261.6, 329.6, 392], 1.5, 0.1, 1500);
      s.tone({ at: s.t + 0.3, freq: 130.8, dur: 1.5, vol: 0.25, shape: 'hold', attack: 0.05, release: 0.6 });
    },
  },
  'mission.failed': {
    reverb: 0.4,
    build: (s) => {
      const lp = s.filter('lowpass', 900, 1);
      const notes = [440, 415.3, 349.2];
      notes.forEach((f, i) => s.tone({ at: s.t + i * 0.3, freq: f, dur: 0.9, vol: 0.2, type: 'sawtooth', shape: 'hold', attack: 0.02, release: 0.4, dest: lp }));
      s.tone({ at: s.t + 0.6, freq: 55, glideTo: 35, dur: 1.2, vol: 0.4, attack: 0.02 });
    },
  },
  'mission.timer.tick': {
    build: (s) => {
      s.click(s.t, 0.25, 2200, 4, 0.01);
      s.tone({ freq: 1000, dur: 0.04, vol: 0.15 });
    },
  },
  'cutscene.skip': {
    build: (s) => {
      s.noiseBurst({ dur: 0.18, vol: 0.12, filter: 'bandpass', freq: 600, freqTo: 2400, q: 1, attack: 0.03 });
      s.tone({ at: s.t + 0.08, freq: 1320, dur: 0.1, vol: 0.2 });
    },
  },
  'dialogue.tick': { build: (s) => s.click(s.t, 0.12, 2600, 5, 0.01) },
  'ability.activate': {
    reverb: 0.3,
    build: (s) => {
      const lp = s.filter('lowpass', 1800, 1.5);
      s.tone({ freq: 110, glideTo: 440, dur: 0.5, vol: 0.3, type: 'sawtooth', shape: 'hold', attack: 0.02, release: 0.3, dest: lp });
      s.thump(s.t + 0.05, 60, 40, 0.3, 0.5);
      s.thump(s.t + 0.4, 60, 40, 0.3, 0.4);
    },
  },
  'ability.end': {
    build: (s) => {
      const lp = s.filter('lowpass', 1500, 1.2);
      s.tone({ freq: 440, glideTo: 110, dur: 0.45, vol: 0.28, type: 'sawtooth', shape: 'hold', attack: 0.01, release: 0.25, dest: lp });
    },
  },
  'ability.recharged': { reverb: 0.2, build: (s) => s.arp(s.t, [880, 1108, 1320], 0.07, 0.25, 0.22) },
  'character.switch': {
    build: (s) => {
      s.whoosh(s.t, 300, 2500, 0.3, 0.3);
      s.tone({ at: s.t + 0.2, freq: 1320, dur: 0.12, vol: 0.2 });
    },
  },
  'character.switch.whoosh': {
    reverb: 0.3,
    build: (s) => {
      s.noiseBurst({ dur: 0.45, vol: 0.4, filter: 'bandpass', freq: 200, freqTo: 3000, q: 1, attack: 0.15, shape: 'hold', release: 0.05 });
      s.noiseBurst({ at: s.t + 0.45, dur: 0.45, vol: 0.4, filter: 'bandpass', freq: 3000, freqTo: 200, q: 1, attack: 0.02 });
    },
  },
};
