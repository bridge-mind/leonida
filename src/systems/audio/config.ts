import type { AudioBus } from './api';

/** All audio tunables. Units: seconds, Hz, meters, linear gain 0..1. */
export const AudioConfig = {
  storageKey: 'leonida.audio',

  voices: {
    maxOneShots: 40,
    maxLoops: 16,
    /** One-shots farther than this from the listener are not scheduled. */
    maxDistance: 220,
    /** `play()` of a `.loop` id plays it for this long, then fades. */
    playLoopSeconds: 1.5,
    stopFade: 0.08,
    /** Time constant for `setLoopParams` ramps. */
    paramRamp: 0.04,
    /** How far ahead periodic loops schedule their events. */
    lookahead: 0.2,
  },

  panner: {
    model: 'equalpower' as PanningModelType,
    distanceModel: 'inverse' as DistanceModelType,
    refDistance: 6,
    maxDistance: 220,
    rolloff: 1.2,
  },

  doppler: {
    speedOfSound: 340,
    /** Maximum pitch shift (fraction) applied to moving positional loops. */
    maxShift: 0.12,
    smoothing: 0.12,
  },

  defaultBusVolumes: { master: 0.9, sfx: 1, engine: 0.8, ambient: 0.7, ui: 0.8, music: 0.6 } as Record<AudioBus, number>,

  compressor: { threshold: -14, knee: 18, ratio: 4, attack: 0.003, release: 0.25 },

  reverb: { seconds: 1.6, decay: 4, wet: 0.16 },

  noise: { seconds: 2 },

  footsteps: {
    minSpeed: 0.5,
    strideWalk: 0.75,
    strideSprint: 1.4,
    walkSpeed: 2,
    sprintSpeed: 7,
    surfaceRayInterval: 0.5,
    rayHeight: 0.6,
    rayLength: 1.6,
    volumeMin: 0.3,
    volumeMax: 0.7,
    pitchJitter: 0.1,
    /** Fall speeds (m/s) that trigger a landing sound and the hard variant. */
    landMinFallSpeed: 2.5,
    hardLandFallSpeed: 11,
  },

  cues: {
    /** Minimum seconds between two `player.hurt` cues. */
    hurtInterval: 0.25,
  },

  ambient: {
    pollInterval: 0.5,
    crossfade: 2,
    bedVolume: 0.55,
    cricketsMax: 0.35,
    fallback: 'ambient.night.crickets.loop',
    cityFallback: 'ambient.city.loop',
    interior: 'ambient.interior.loop',
    casino: 'casino.ambient.loop',
    /** Districts where the night crickets bed stays silent. */
    noCricketDistricts: ['downtown', 'viceBeach'] as readonly string[],
  },

  engine: {
    /** Lower speed bound (m/s) of gears 1..5. */
    gearSpeeds: [0, 9, 17, 27, 38] as readonly number[],
    shiftSeconds: 0.18,
    shiftDip: 0.7,
    /** Frequency multiplier range: idle × (1 + rpmRange) at full rpm. */
    rpmRange: 2.5,
    ramp: 0.045,
    /** Blend between the caller's rpm01 and the in-gear speed fraction. */
    gearBlend: 0.4,
  },

  radio: { bpm: 120, steps: 16, volume: 0.5 },

  debug: { playAllStaggerMs: 30 },
};
