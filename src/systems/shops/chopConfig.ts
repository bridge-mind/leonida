export const ChopShopConfig = {
  /** Radius of the drive-in volume in front of the garage door. */
  radius: 6,
  /** How far in front of the door the volume centre sits. */
  forward: 5,
  /** Vehicle must be slower than this (m/s) to get the prompt. */
  maxSpeed: 1,
  /** Seconds of black while the car is "processed". */
  holdSeconds: 0.8,
  /** Archetypes that never appear as the daily wanted car. */
  excluded: ['police', 'swat'] as readonly string[],
} as const;
