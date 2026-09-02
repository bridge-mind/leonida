/** Pickup tunables. Units: meters, seconds. */
export const PickupConfig = {
  maxLive: 64,
  /** Capacity per kind in the instanced renderer (weapon + ammo share a mesh). */
  capacityPerKind: 64,
  collectRadius: 1.2,
  magnetRadius: 2.5,
  magnetSpeed: 6,
  lifetime: 60,
  /** Height of the pickup's center above the ground. */
  hoverHeight: 0.45,
  bobAmplitude: 0.08,
  bobFrequency: 2.2,
  spinSpeed: 1.6,
  /** Fade-out blink starts this many seconds before expiry. */
  blinkBefore: 8,
} as const;
