import type { Recipe, RecipeTable } from './types';

/** Soft blip: 5–20 ms attack, 80–200 ms decay. */
function blip(freq: number, dur = 0.1, vol = 0.25, type: OscillatorType = 'sine'): Recipe {
  return { build: (s) => s.tone({ freq, dur, vol, type, attack: 0.008 }) };
}

function twoNote(a: number, b: number, step = 0.07, dur = 0.12, vol = 0.25, type: OscillatorType = 'sine', reverb = 0): Recipe {
  return { reverb, build: (s) => s.arp(s.t, [a, b], step, dur, vol, type) };
}

function clickTone(clickHz: number, toneHz: number, toneDur: number, vol: number): Recipe {
  return {
    build: (s) => {
      s.click(s.t, vol, clickHz, 4, 0.012);
      s.tone({ freq: toneHz, dur: toneDur, vol: vol * 0.8, attack: 0.002 });
    },
  };
}

export const UI_RECIPES: RecipeTable<'ui.'> = {
  'ui.blip': blip(1200, 0.08),
  'ui.select': twoNote(880, 1320, 0.06),
  'ui.back': twoNote(1320, 880, 0.06),
  'ui.open': {
    build: (s) => {
      s.tone({ freq: 660, glideTo: 990, dur: 0.16, vol: 0.25, type: 'triangle', attack: 0.01 });
      s.noiseBurst({ dur: 0.12, vol: 0.08, filter: 'bandpass', freq: 800, freqTo: 2400, q: 1 });
    },
  },
  'ui.close': {
    build: (s) => {
      s.tone({ freq: 990, glideTo: 660, dur: 0.16, vol: 0.25, type: 'triangle', attack: 0.01 });
      s.noiseBurst({ dur: 0.12, vol: 0.08, filter: 'bandpass', freq: 2400, freqTo: 800, q: 1 });
    },
  },
  'ui.error': {
    build: (s) => {
      const lp = s.filter('lowpass', 600, 1);
      s.tone({ freq: 110, dur: 0.3, vol: 0.3, type: 'square', shape: 'hold', attack: 0.01, release: 0.04, dest: lp });
      s.tone({ freq: 115, dur: 0.3, vol: 0.2, type: 'square', shape: 'hold', attack: 0.01, release: 0.04, dest: lp });
    },
  },
  'ui.notification': twoNote(988, 1319, 0.09, 0.25, 0.25, 'sine', 0.2),
  'ui.notification.money': { reverb: 0.15, build: (s) => s.arp(s.t, [1175, 1568, 1976], 0.05, 0.22, 0.25) },
  'ui.wheel.open': {
    build: (s) => {
      s.noiseBurst({ dur: 0.25, vol: 0.12, filter: 'bandpass', freq: 300, freqTo: 1500, q: 1, attack: 0.05 });
      s.tone({ freq: 440, glideTo: 880, dur: 0.2, vol: 0.2, type: 'triangle', attack: 0.01 });
    },
  },
  'ui.wheel.tick': clickTone(2500, 1800, 0.03, 0.2),
  'ui.wheel.select': {
    build: (s) => {
      s.tone({ freq: 660, glideTo: 990, dur: 0.12, vol: 0.25, attack: 0.005 });
      s.click(s.t, 0.2, 3000, 4, 0.01);
    },
  },
  'ui.map.open': {
    build: (s) => {
      s.noiseBurst({ dur: 0.2, vol: 0.1, filter: 'bandpass', freq: 500, freqTo: 2000, q: 1, attack: 0.04 });
      s.tone({ freq: 523, dur: 0.2, vol: 0.2, type: 'triangle', attack: 0.01 });
    },
  },
  'ui.map.pin': clickTone(2000, 1500, 0.08, 0.25),
  'ui.pause': {
    build: (s) => {
      s.tone({ freq: 440, dur: 0.14, vol: 0.22, type: 'triangle', shape: 'hold', release: 0.04 });
      s.tone({ at: s.t + 0.15, freq: 330, dur: 0.18, vol: 0.22, type: 'triangle', shape: 'hold', release: 0.05 });
    },
  },
  'ui.unpause': {
    build: (s) => {
      s.tone({ freq: 330, dur: 0.14, vol: 0.22, type: 'triangle', shape: 'hold', release: 0.04 });
      s.tone({ at: s.t + 0.15, freq: 440, dur: 0.18, vol: 0.22, type: 'triangle', shape: 'hold', release: 0.05 });
    },
  },
  'ui.phone.open': {
    build: (s) => {
      s.click(s.t, 0.2, 2500, 4, 0.01);
      s.arp(s.t + 0.02, [1568, 2093], 0.05, 0.1, 0.18);
    },
  },
  'ui.phone.close': {
    build: (s) => {
      s.arp(s.t, [2093, 1568], 0.05, 0.1, 0.18);
      s.click(s.t + 0.1, 0.2, 2000, 4, 0.01);
    },
  },
  'ui.phone.ring.loop': {
    build: (s) => {
      s.periodic(
        () => 2,
        (at) => {
          for (const off of [0, 0.25]) {
            s.tone({ at: at + off, freq: 1200, dur: 0.12, vol: 0.2, shape: 'hold', attack: 0.004, release: 0.02 });
            s.tone({ at: at + off, freq: 1600, dur: 0.12, vol: 0.15, shape: 'hold', attack: 0.004, release: 0.02 });
          }
        },
      );
    },
  },
  'ui.phone.text': twoNote(1760, 2349, 0.12, 0.1, 0.2),
  'ui.phone.hangup': {
    build: (s) => {
      s.click(s.t, 0.25, 1800, 4, 0.012);
      s.tone({ at: s.t + 0.02, freq: 400, dur: 0.15, vol: 0.2, shape: 'hold', release: 0.03 });
    },
  },
  'ui.snapmatic': {
    build: (s) => {
      s.click(s.t, 0.45, 3000, 3, 0.01);
      s.noiseBurst({ at: s.t + 0.01, dur: 0.04, vol: 0.25, filter: 'bandpass', freq: 1200, q: 1.5 });
      s.click(s.t + 0.06, 0.4, 1800, 3, 0.015);
    },
  },
  'ui.hitmarker': clickTone(2800, 1400, 0.04, 0.25),
  'ui.hitmarker.kill': clickTone(2000, 900, 0.09, 0.32),
  'ui.save': { reverb: 0.15, build: (s) => s.arp(s.t, [880, 1320, 1760], 0.05, 0.15, 0.22, 'triangle') },
  'ui.load': { reverb: 0.15, build: (s) => s.arp(s.t, [1760, 1320, 880], 0.05, 0.15, 0.22, 'triangle') },
  'ui.title.stinger': {
    reverb: 0.5,
    build: (s) => {
      const lp = s.filter('lowpass', 1200, 0.8);
      const notes = [220, 277.2, 329.6, 440];
      notes.forEach((f, i) => {
        const at = s.t + i * 0.4;
        s.tone({ at, freq: f, dur: 2.4 - i * 0.3, vol: 0.14, type: 'sawtooth', shape: 'hold', attack: 0.15, release: 1.4, detune: -7, dest: lp });
        s.tone({ at, freq: f, dur: 2.4 - i * 0.3, vol: 0.14, type: 'sawtooth', shape: 'hold', attack: 0.15, release: 1.4, detune: 7, dest: lp });
      });
      s.tone({ freq: 55, dur: 3.4, vol: 0.3, shape: 'hold', attack: 0.3, release: 1.2 });
    },
  },
};
