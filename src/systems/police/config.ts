/** Police-system tunables. Units: meters, seconds, radians. */
export const PoliceConfig = {
  dispatch: {
    /** Seconds between dispatch checks (spawning missing units). */
    interval: 1.5,
    /** Extra delay after the level first rises before the first unit spawns; shrinks with heat. */
    responseDelay: 4,
    responseDelayMinAtMaxHeat: 1,
    spawnMin: 150,
    spawnMax: 250,
    /** Lane point must be at least this far from other vehicles. */
    spawnClearance: 8,
    /** Attempts per dispatch tick to find a clear, off-screen lane point. */
    spawnTries: 6,
    /** Hard cap on cruisers + vans + helicopters + roadblocks. */
    maxUnits: 12,
    maxCops: 28,
    /** Units farther than this and off-screen are recycled (new ones spawn closer). */
    recycleDistance: 320,
    /** Seconds a stood-down unit keeps driving/standing before it is removed. */
    leaveSeconds: 8,
    leaveHardCap: 20,
  },
  driver: {
    repathInterval: 2,
    nodeReach: 9,
    laneOffset: 3.2,
    steerGain: 1.6,
    cruiseSpeed: 26,
    turnSlowAngle: 0.6,
    turnSpeed: 9,
    pursuitRange: 40,
    /** Player on foot inside this range: stop and let the cops out. */
    exitRange: 25,
    exitSpeed: 2,
    /** Predicts the player's car this far ahead when ramming. */
    ramLead: 0.5,
    /** Non-ramming tiers keep this gap behind the player's car. */
    followGap: 9,
    losInterval: 0.25,
    losRange: 90,
    stuckSpeed: 0.6,
    stuckAfter: 2,
    reverseSeconds: 1.3,
    throttleGain: 0.35,
    brakeGain: 0.25,
  },
  cop: {
    eyeHeight: 1.6,
    chestHeight: 1.2,
    sightRange: 60,
    losInterval: 0.3,
    arrestRange: 2.5,
    /** Seconds a cop must stay within arrest range with LOS while the player stands still. */
    arrestSeconds: 1.0,
    /** Player speed above which they count as fleeing (no arrest attempt, cops shoot). */
    fleeSpeed: 2.5,
    /** Stopped-vehicle speed below which the driver can be pulled out and busted. */
    vehicleStillSpeed: 1,
    approachWalkDistance: 15,
    chaseDistance: 30,
    standoffMin: 12,
    standoffMax: 20,
    strafeSpeed: 2.2,
    strafeSwitchMin: 1.2,
    strafeSwitchMax: 2.8,
    backOffSpeed: 2.4,
    accuracyBase: 0.35,
    accuracySwat: 0.6,
    /** accuracy += heat / heatDivisor. */
    heatDivisor: 300,
    /** Firing cadence jitter multiplier range. */
    cadenceJitter: 0.35,
    /** Grace period after a cop first acquires LOS before the first shot, so the player can react. */
    firstShotDelay: 0.6,
  },
  helicopter: {
    altitude: 45,
    orbitRadius: 35,
    orbitSpeed: 0.28,
    /** Horizontal chase speed toward the orbit point. */
    speed: 28,
    health: 600,
    fireInterval: 0.4,
    accuracy: 0.3,
    losRange: 250,
    losInterval: 0.25,
    rotorSpeed: 30,
    spotlightIntensity: 900,
    spotlightAngle: 0.22,
    /** Rotor sound loop pitch. */
    loopPitch: 1.05,
    /** Seconds the wreck falls before it is removed. */
    crashSeconds: 3,
    fallSpeed: 12,
    explosionRadius: 9,
    explosionDamage: 220,
    explosionImpulse: 900,
  },
  roadblock: {
    aheadMin: 120,
    aheadMax: 200,
    /** Roadblocks are rebuilt when the player gets farther than this. */
    staleDistance: 320,
    cooldown: 25,
    /** Cruisers are placed this far to either side of the road center. */
    lateralSpacing: 3.4,
    spikeHalfLength: 6,
    spikeDamage: 150,
    spikeImpulse: 6,
    /** A car re-triggers the spike after this many seconds. */
    spikeCooldown: 3,
    minSpeedForSpike: 4,
  },
} as const;
