/** Wanted-system tunables. Units: meters, seconds. */
export const WantedConfig = {
  witness: {
    /** Civilian witness radius for quiet / loud crimes. */
    civilianRadius: 25,
    civilianLoudRadius: 60,
    /** A cop this close sees the crime directly (no phone call, state goes `active`). */
    copRadius: 40,
    copLoudRadius: 60,
    /** Ped queries are horizontal; interiors sit 600 m above the city, so a witness must also be on the same level. */
    maxVerticalGap: 12,
  },
  vision: {
    /** Seconds between line-of-sight sweeps. */
    interval: 0.25,
    /** Raycasts per sweep (round-robin over the cops in range). */
    raysPerCheck: 6,
    /** `playerSeen` is true this long after the last hit. */
    seenWindow: 1.0,
    /** Without a hit for this long, `active` decays to `searching`. */
    loseSightAfter: 3.0,
    footRange: 60,
    vehicleRange: 90,
    helicopterRange: 250,
    eyeHeight: 1.6,
    /** Eye of a cop inside a cruiser sits above the roof so the ray clears its own body. */
    vehicleEyeHeight: 1.75,
    chestHeight: 1.2,
    vehicleTargetHeight: 1.0,
    /** A cruiser this close to the player counts as seeing them without a ray. */
    nearAutoSee: 6,
  },
  search: {
    initialRadius: 40,
    growthPerSecond: 8,
    maxRadiusBase: 220,
    maxRadiusPerLevel: 30,
  },
  evasion: {
    baseSeconds: 20,
    perLevelSeconds: 20,
    /** Hidden inside the circle counts this much slower than being outside it. */
    hiddenInsideMultiplier: 1.5,
  },
  hotScene: {
    radius: 60,
    expirySeconds: 120,
    /** A new crime this close to an existing scene refreshes it instead of adding one. */
    mergeRadius: 30,
    max: 8,
    reraiseLevel: 1,
  },
  heat: {
    perStar: 5,
    decayPerMinute: 1,
    max: 100,
  },
  collision: {
    /** `vehicle:collided` impulse above which ramming a cruiser is a crime. */
    copVehicleImpulse: 3,
  },
} as const;
