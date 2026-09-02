import type { SoundId } from '../sounds';
import type { Synth } from '../synth';
import type { Recipe } from './types';

interface Boom {
  dur: number;
  lpFrom: number;
  lpTo: number;
  boomHz: number;
  boomDur: number;
  vol: number;
  crack: boolean;
  rumbleDelay: number;
  rumbleDur: number;
}

/** Brown-noise burst with a long lowpass decay, a sub boom, an optional crack, and a delayed rumble tail. */
export function explosion(s: Synth, b: Boom): void {
  s.noiseBurst({ kind: 'brown', dur: b.dur, vol: b.vol, filter: 'lowpass', freq: b.lpFrom, freqTo: b.lpTo, q: 0.7, attack: 0.004, sweepTime: b.dur });
  s.tone({ freq: b.boomHz, glideTo: b.boomHz * 0.55, dur: b.boomDur, vol: b.vol * 0.9, attack: 0.005 });
  if (b.crack) s.noiseBurst({ dur: 0.05, vol: b.vol * 0.7, filter: 'highpass', freq: 1800, attack: 0.001 });
  s.noiseBurst({ kind: 'brown', at: s.t + b.rumbleDelay, dur: b.rumbleDur, vol: b.vol * 0.45, filter: 'lowpass', freq: 220, freqTo: 60, attack: 0.1 });
}

/** Crackling fire: lowpassed noise, a rumble and random high crackle ticks. */
export function buildFire(s: Synth, size: number): void {
  const g = s.gain(0.5 * size);
  const lp = s.filter('lowpass', 420, 0.7, g);
  s.noiseSrc('brown', lp);
  s.lfo(0.7, 0.12, g.gain);
  const hiss = s.gain(0.12 * size, g);
  const hissBp = s.filter('bandpass', 2600, 0.9, hiss);
  s.noiseSrc('white', hissBp);
  s.lfo(2.3, 0.06, hiss.gain);
  const rumble = s.gain(0.15 * size, g);
  const rumbleLp = s.filter('lowpass', 120, 1, rumble);
  s.osc('sawtooth', 42, rumbleLp);
  s.periodic(
    () => s.rand(0.04, 0.22),
    (at) => s.click(at, s.rand(0.1, 0.3) * size, s.rand(1800, 5000), 4, s.rand(0.006, 0.02)),
  );
}

type ExplosionId = Extract<SoundId, `explosion.${string}`> | 'fire.loop';

export const EXPLOSION_RECIPES: Record<ExplosionId, Recipe> = {
  'explosion.large': {
    reverb: 0.5,
    build: (s) => explosion(s, { dur: 1.8, lpFrom: 3200, lpTo: 140, boomHz: 46, boomDur: 1.3, vol: 1, crack: true, rumbleDelay: 0.3, rumbleDur: 2.2 }),
  },
  'explosion.small': {
    reverb: 0.35,
    build: (s) => explosion(s, { dur: 0.7, lpFrom: 3000, lpTo: 200, boomHz: 60, boomDur: 0.5, vol: 0.8, crack: true, rumbleDelay: 0.15, rumbleDur: 0.9 }),
  },
  'explosion.debris': {
    build: (s) => {
      for (let i = 0; i < 9; i++) {
        const at = s.t + s.rand(0.05, 0.9);
        if (i % 3 === 0) s.ping(at, s.rand(700, 1800), 0.15, 0.25, 15);
        else s.thump(at, s.rand(120, 260), 80, 0.08, 0.3);
        s.click(at, 0.25, s.rand(800, 3000), 2, 0.02);
      }
    },
  },
  'explosion.distant': {
    reverb: 0.6,
    build: (s) => explosion(s, { dur: 2.5, lpFrom: 320, lpTo: 60, boomHz: 36, boomDur: 2, vol: 0.7, crack: false, rumbleDelay: 0.5, rumbleDur: 2.6 }),
  },
  'fire.loop': { build: (s) => buildFire(s, 1) },
};
