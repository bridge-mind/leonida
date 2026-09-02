import type { Synth } from '../synth';
import type { Recipe, RecipeTable } from './types';

interface Gunshot {
  thumpHz: number;
  thumpDur: number;
  noiseDur: number;
  lpFrom: number;
  lpTo: number;
  vol: number;
  crack?: boolean;
  tail?: number;
  reverb?: number;
}

/** White-noise burst through a sweeping lowpass plus a low "thump"; optional high crack and brown tail. */
function gunshot(g: Gunshot): Recipe {
  return {
    reverb: g.reverb ?? 0.2,
    build: (s) => {
      s.noiseBurst({ dur: g.noiseDur, vol: g.vol, filter: 'lowpass', freq: g.lpFrom, freqTo: g.lpTo, q: 0.8, attack: 0.001, sweepTime: g.noiseDur });
      s.thump(s.t, g.thumpHz, g.thumpHz * 0.45, g.thumpDur, g.vol * 0.9);
      if (g.crack) s.noiseBurst({ dur: 0.012, vol: g.vol * 0.7, filter: 'highpass', freq: 3200, attack: 0.0005 });
      if (g.tail) s.noiseBurst({ kind: 'brown', dur: g.tail, vol: g.vol * 0.5, filter: 'lowpass', freq: 900, freqTo: 150, attack: 0.005 });
    },
  };
}

/** Mechanical reload: clicks at `offsets` (seconds), optional slide noises. */
function reload(offsets: readonly number[], freq: number, slides: readonly number[] = []): Recipe {
  return {
    build: (s) => {
      for (const o of offsets) s.click(s.t + o, 0.5, s.jitter(freq, 0.08), 4, 0.02);
      for (const o of slides) s.noiseBurst({ at: s.t + o, dur: 0.09, vol: 0.25, filter: 'lowpass', freq: 1400, q: 0.7, attack: 0.02 });
    },
  };
}

function dryClick(freq: number): Recipe {
  return {
    build: (s) => {
      s.click(s.t, 0.45, freq, 6, 0.012);
      s.click(s.t + 0.03, 0.2, freq * 0.7, 5, 0.01);
    },
  };
}

function whooshHit(s: Synth, thumpHz: number, dur: number, vol: number): void {
  s.thump(s.t, thumpHz, thumpHz * 0.5, dur, vol);
  s.noiseBurst({ dur: dur * 0.5, vol: vol * 0.6, filter: 'lowpass', freq: 900, freqTo: 200, attack: 0.002 });
}

export const WEAPON_RECIPES: RecipeTable<'weapon.'> = {
  'weapon.pistol.fire': gunshot({ thumpHz: 140, thumpDur: 0.12, noiseDur: 0.022, lpFrom: 6000, lpTo: 400, vol: 0.9 }),
  'weapon.pistol.reload': reload([0, 0.12, 0.36], 1800),
  'weapon.pistol.empty': dryClick(2400),
  'weapon.smg.fire': gunshot({ thumpHz: 170, thumpDur: 0.06, noiseDur: 0.012, lpFrom: 5000, lpTo: 600, vol: 0.7, reverb: 0.12 }),
  'weapon.smg.reload': reload([0, 0.15, 0.4, 0.5], 2000, [0.2]),
  'weapon.smg.empty': dryClick(2800),
  'weapon.shotgun.fire': gunshot({ thumpHz: 70, thumpDur: 0.35, noiseDur: 0.04, lpFrom: 4000, lpTo: 300, vol: 1, tail: 0.18, reverb: 0.3 }),
  'weapon.shotgun.reload': reload([0.1, 0.36], 1100, [0, 0.26]),
  'weapon.shotgun.empty': dryClick(1600),
  'weapon.rifle.fire': gunshot({ thumpHz: 110, thumpDur: 0.15, noiseDur: 0.025, lpFrom: 6000, lpTo: 500, vol: 0.85, crack: true }),
  'weapon.rifle.reload': reload([0, 0.2, 0.55, 0.62], 1500, [0.3]),
  'weapon.rifle.empty': dryClick(2100),
  'weapon.sniper.fire': gunshot({ thumpHz: 55, thumpDur: 0.5, noiseDur: 0.035, lpFrom: 7000, lpTo: 300, vol: 1, crack: true, tail: 0.3, reverb: 0.45 }),
  'weapon.sniper.reload': reload([0, 0.4, 0.72], 1000, [0.12, 0.5]),
  'weapon.sniper.empty': dryClick(1400),
  'weapon.rpg.fire': {
    reverb: 0.3,
    build: (s) => {
      s.thump(s.t, 60, 30, 0.4, 0.8);
      s.noiseBurst({ dur: 0.08, vol: 0.9, filter: 'lowpass', freq: 3000, freqTo: 400, attack: 0.002 });
      // Rocket motor receding: bandpassed noise sweeping down while fading.
      s.noiseBurst({ at: s.t + 0.05, dur: 0.9, vol: 0.5, filter: 'bandpass', freq: 1600, freqTo: 250, q: 1.5, attack: 0.03 });
    },
  },
  'weapon.rpg.reload': reload([0, 0.5, 0.9], 800, [0.2]),
  'weapon.grenade.throw': { build: (s) => s.whoosh(s.t, 400, 1300, 0.25, 0.35) },
  'weapon.grenade.bounce': {
    build: (s) => {
      s.ping(s.t, 900, 0.15, 0.35, 18);
      s.thump(s.t, 220, 120, 0.08, 0.3);
    },
  },
  'weapon.melee.swing': { build: (s) => s.whoosh(s.t, 500, 2600, 0.18, 0.35, 1.5) },
  'weapon.melee.hit': { build: (s) => whooshHit(s, 120, 0.12, 0.6) },
  'weapon.melee.knockdown': {
    build: (s) => {
      whooshHit(s, 80, 0.25, 0.8);
      s.noiseBurst({ kind: 'brown', dur: 0.15, vol: 0.5, filter: 'lowpass', freq: 400, attack: 0.005 });
    },
  },
  'weapon.switch': {
    build: (s) => {
      s.click(s.t, 0.4, 1500, 4);
      s.click(s.t + 0.06, 0.35, 2200, 4);
    },
  },
  'weapon.cock': {
    build: (s) => {
      s.click(s.t, 0.5, 1200, 4, 0.03);
      s.noiseBurst({ at: s.t + 0.03, dur: 0.06, vol: 0.2, filter: 'lowpass', freq: 1500, attack: 0.01 });
      s.click(s.t + 0.09, 0.45, 2000, 4);
    },
  },
  'weapon.dryfire': dryClick(2600),
  'weapon.shell.drop': {
    build: (s) => {
      s.ping(s.t, s.jitter(4200, 0.1), 0.15, 0.25, 40);
      s.ping(s.t + 0.08, s.jitter(5100, 0.1), 0.1, 0.15, 40);
    },
  },
};
