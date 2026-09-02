/** Particle system tunables. */
export const ParticlesConfig = {
  /** Pool capacity of the additive layer (sparks, flash, fire, fireball). */
  additiveCapacity: 2048,
  /** Pool capacity of the normal-blended layer (smoke, dust, blood, shards). */
  normalCapacity: 2048,
  tracerCapacity: 128,
  /** Seconds a tracer takes to fade out. */
  tracerLife: 0.07,
  tracerColor: 0xffe2a0,
  muzzleFlash: { duration: 0.04, lightIntensity: 14, lightDistance: 9, spriteSize: 0.45 },
  /** Particles below the ground plane by more than this are killed early (keeps dust from sinking). */
  floorY: -0.5,
};
