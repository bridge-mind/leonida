/** Damage system tunables. */
export const DamageConfig = {
  headshotMultiplier: 3,
  explosion: {
    /** Fraction of the radius that receives full damage before the linear falloff starts. */
    fullDamageFraction: 1 / 3,
    /** Ped ragdoll impulse per unit of `impulse`: horizontal and vertical factors. */
    pedImpulseForward: 0.02,
    pedImpulseUp: 0.01,
    /** Camera shake at zero distance; scales down linearly to zero at `shakeRange`. */
    shakeStrength: 0.5,
    shakeSeconds: 0.6,
    shakeRange: 60,
    /** Particle scale = radius / this. */
    particleScaleDivisor: 6,
  },
};
