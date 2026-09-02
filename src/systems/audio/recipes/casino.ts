import type { Synth } from '../synth';
import { voiceLine } from './peds';
import type { Recipe, RecipeTable } from './types';

/** Short pitched chip click with slight randomisation. */
function chip(s: Synth, at: number, vol: number): void {
  s.click(at, vol, s.rand(2300, 2900), 5, 0.01);
  s.tone({ at, freq: s.rand(2900, 3500), dur: 0.05, vol: vol * 0.5, attack: 0.001 });
}

/** Paper swish: bandpassed noise sweeping upward. */
function swish(s: Synth, at: number, dur: number, vol: number, from = 1800, to = 4000): void {
  s.noiseBurst({ at, dur, vol, filter: 'bandpass', freq: s.jitter(from, 0.1), freqTo: to, q: 1.2, attack: dur * 0.3 });
}

function bell(s: Synth, at: number, freq: number, vol: number): void {
  s.ping(at, freq, 0.5, vol, 40);
  s.tone({ at, freq: freq * 2.4, dur: 0.3, vol: vol * 0.3, attack: 0.001 });
}

function sting(freqs: readonly number[], step: number, dur: number, vol: number, type: OscillatorType = 'triangle', reverb = 0.25): Recipe {
  return { reverb, build: (s) => s.arp(s.t, freqs, step, dur, vol, type) };
}

const MAJOR_RUN = [523.3, 587.3, 659.3, 784, 880, 1046.5, 1174.7, 1318.5, 1568, 1760, 2093, 2637];

export const CASINO_RECIPES: RecipeTable<'casino.'> = {
  'casino.ambient.loop': {
    build: (s) => {
      const g = s.gain(0.25);
      const lp = s.filter('lowpass', 800, 0.7, g);
      s.noiseSrc('pink', lp);
      const murmur = s.filter('bandpass', 700, 2, g);
      let nextChip = 0;
      s.periodic(
        () => s.rand(0.12, 0.3),
        (at) => {
          const f = s.rand(140, 300);
          voiceLine(s, at, { f0: f, f1: f * s.rand(0.9, 1.2), f2: f * s.rand(0.8, 1.05), dur: s.rand(0.08, 0.18), formant: s.rand(600, 1100), vol: 0.09, dest: murmur });
          if (at >= nextChip) {
            chip(s, at, 0.08);
            if (Math.random() < 0.5) s.ping(at + 0.03, s.rand(3500, 4500), 0.12, 0.05, 30);
            nextChip = at + s.rand(1, 4);
          }
        },
      );
    },
  },
  'casino.cashier': {
    reverb: 0.15,
    build: (s) => {
      s.click(s.t, 0.35, 900, 3, 0.02);
      bell(s, s.t + 0.05, 2600, 0.3);
      chip(s, s.t + 0.2, 0.25);
    },
  },
  'casino.chip.place': { build: (s) => chip(s, s.t, 0.4) },
  'casino.chip.collect': {
    build: (s) => {
      for (let i = 0; i < 4; i++) chip(s, s.t + i * s.rand(0.03, 0.06), 0.3);
    },
  },
  'casino.chip.stack': {
    build: (s) => {
      chip(s, s.t, 0.35);
      chip(s, s.t + 0.05, 0.3);
    },
  },
  'casino.chip.clink': { build: (s) => s.ping(s.t, s.jitter(4000, 0.12), 0.12, 0.3, 25) },
  'casino.card.deal': { build: (s) => swish(s, s.t, 0.06, 0.3) },
  'casino.card.flip': {
    build: (s) => {
      swish(s, s.t, 0.05, 0.25, 1500, 3500);
      swish(s, s.t + 0.06, 0.04, 0.2, 2500, 5000);
    },
  },
  'casino.card.shuffle': {
    build: (s) => {
      for (let i = 0; i < 8; i++) swish(s, s.t + i * 0.06, 0.05, 0.22, s.rand(1400, 2200), s.rand(3000, 4500));
    },
  },
  'casino.card.slide': { build: (s) => swish(s, s.t, 0.15, 0.25, 900, 2200) },
  'casino.blackjack': sting([659.3, 784, 1046.5, 1318.5], 0.08, 0.4, 0.28),
  'casino.bust': {
    reverb: 0.2,
    build: (s) => {
      const lp = s.filter('lowpass', 1200, 1);
      s.arp(s.t, [440, 349.2, 261.6], 0.12, 0.35, 0.2, 'sawtooth');
      s.tone({ at: s.t + 0.3, freq: 110, dur: 0.3, vol: 0.2, type: 'square', shape: 'hold', dest: lp });
    },
  },
  'casino.push': sting([660, 660], 0.15, 0.25, 0.22, 'sine', 0.15),
  'casino.reel.spin.loop': {
    build: (s) => {
      s.periodic(
        () => 0.06 / s.currentPitch,
        (at) => s.click(at, 0.25, s.jitter(1500, 0.05), 4, 0.01),
      );
    },
  },
  'casino.reel.stop': {
    build: (s) => {
      s.click(s.t, 0.4, 800, 3, 0.02);
      s.thump(s.t, 200, 120, 0.06, 0.3);
    },
  },
  'casino.slot.lever': {
    build: (s) => {
      s.click(s.t, 0.35, 700, 3, 0.02);
      s.noiseBurst({ at: s.t + 0.02, dur: 0.15, vol: 0.2, filter: 'lowpass', freq: 900, attack: 0.03, shape: 'hold' });
      s.thump(s.t + 0.17, 150, 90, 0.08, 0.45);
      s.click(s.t + 0.17, 0.35, 1400, 3, 0.02);
    },
  },
  'casino.slot.coin': {
    build: (s) => {
      s.ping(s.t, 3500, 0.15, 0.3, 30);
      s.ping(s.t + 0.05, 4200, 0.12, 0.25, 30);
    },
  },
  'casino.slot.win': {
    reverb: 0.25,
    build: (s) => {
      s.arp(s.t, [523.3, 659.3, 784, 1046.5], 0.08, 0.3, 0.25, 'triangle');
      for (let i = 0; i < 5; i++) s.ping(s.t + 0.3 + i * 0.09, s.rand(3200, 4600), 0.12, 0.2, 30);
    },
  },
  'casino.jackpot': {
    reverb: 0.5,
    build: (s) => {
      s.arp(s.t, MAJOR_RUN, 0.12, 0.45, 0.22, 'triangle');
      for (let i = 0; i < 14; i++) bell(s, s.t + 0.4 + i * 0.16, s.rand(2400, 4800), 0.16);
      const lp = s.filter('lowpass', 1600, 0.8);
      for (const f of [261.6, 329.6, 392, 523.3]) {
        s.tone({ at: s.t + 1.4, freq: f, dur: 1.6, vol: 0.09, type: 'sawtooth', shape: 'hold', attack: 0.1, release: 0.8, detune: -6, dest: lp });
        s.tone({ at: s.t + 1.4, freq: f, dur: 1.6, vol: 0.09, type: 'sawtooth', shape: 'hold', attack: 0.1, release: 0.8, detune: 6, dest: lp });
      }
    },
  },
  'casino.wheel.spin.loop': {
    build: (s) => {
      s.periodic(
        () => 0.12 / s.currentPitch,
        (at) => {
          s.click(at, 0.3, 2000, 4, 0.01);
          s.ping(at, 3200, 0.05, 0.1, 20);
        },
      );
    },
  },
  'casino.wheel.tick': {
    build: (s) => {
      s.click(s.t, 0.3, 2000, 4, 0.01);
      s.ping(s.t, 3200, 0.05, 0.1, 20);
    },
  },
  'casino.wheel.stop': {
    build: (s) => {
      s.click(s.t, 0.4, 1200, 3, 0.02);
      s.thump(s.t, 180, 100, 0.08, 0.3);
    },
  },
  'casino.wheel.prize': {
    reverb: 0.3,
    build: (s) => {
      s.arp(s.t, [784, 988, 1175, 1568], 0.09, 0.35, 0.26, 'triangle');
      bell(s, s.t + 0.4, 3200, 0.25);
    },
  },
  'casino.roulette.spin.loop': {
    build: (s) => {
      const g = s.gain(0.2);
      const bp = s.filter('bandpass', 800, 1, g);
      s.noiseSrc('pink', bp);
      s.lfo(2, 0.08, g.gain);
    },
  },
  'casino.roulette.ball.loop': {
    build: (s) => {
      s.periodic(
        () => s.rand(0.06, 0.1) / s.currentPitch,
        (at) => s.ping(at, s.rand(2600, 3400), 0.04, 0.15, 20),
      );
    },
  },
  'casino.roulette.drop': {
    build: (s) => {
      let at = s.t;
      for (let i = 0; i < 6; i++) {
        s.ping(at, s.rand(2400, 3400), 0.05, 0.25, 20);
        at += 0.16 - i * 0.02;
      }
      s.thump(at, 220, 140, 0.06, 0.3);
      s.click(at, 0.3, 1500, 3, 0.015);
    },
  },
  'casino.roulette.win': {
    reverb: 0.3,
    build: (s) => {
      s.arp(s.t, [659.3, 784, 987.8], 0.1, 0.4, 0.26, 'triangle');
      bell(s, s.t + 0.3, 2800, 0.25);
    },
  },
  'casino.win': sting([659.3, 784, 987.8], 0.1, 0.35, 0.26),
  'casino.win.big': {
    reverb: 0.4,
    build: (s) => {
      s.arp(s.t, [523.3, 659.3, 784, 1046.5, 1318.5, 1568], 0.09, 0.4, 0.24, 'triangle');
      const lp = s.filter('lowpass', 1500, 0.8);
      for (const f of [261.6, 329.6, 392]) {
        s.tone({ at: s.t + 0.5, freq: f, dur: 1.2, vol: 0.09, type: 'sawtooth', shape: 'hold', attack: 0.08, release: 0.6, detune: -5, dest: lp });
        s.tone({ at: s.t + 0.5, freq: f, dur: 1.2, vol: 0.09, type: 'sawtooth', shape: 'hold', attack: 0.08, release: 0.6, detune: 5, dest: lp });
      }
    },
  },
  'casino.lose': {
    build: (s) => {
      const lp = s.filter('lowpass', 1000, 1);
      s.tone({ freq: 392, dur: 0.3, vol: 0.2, type: 'sawtooth', shape: 'hold', release: 0.08, dest: lp });
      s.tone({ at: s.t + 0.25, freq: 329.6, dur: 0.45, vol: 0.2, type: 'sawtooth', shape: 'hold', release: 0.15, dest: lp });
    },
  },
  'casino.bar.pour': {
    build: (s) => {
      s.noiseBurst({ kind: 'pink', dur: 0.8, vol: 0.3, filter: 'bandpass', freq: 1500, freqTo: 600, q: 1.2, attack: 0.05, shape: 'hold', release: 0.15 });
      for (let i = 0; i < 5; i++) s.tone({ at: s.t + s.rand(0.1, 0.8), freq: s.rand(600, 1400), glideTo: 2200, dur: 0.06, vol: 0.12 });
    },
  },
  'casino.guard.alert': {
    build: (s) => {
      voiceLine(s, s.t, { f0: 220, f1: 300, f2: 210, dur: 0.3, formant: 850, vol: 0.5, drive: 1.5 });
      s.tone({ at: s.t + 0.35, freq: 2200, dur: 0.3, vol: 0.25, shape: 'hold', release: 0.05 });
    },
  },
};
