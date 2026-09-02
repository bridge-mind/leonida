/** Tunables for the traffic system (SYSTEM_PROMPT §6). Units: meters, seconds, radians. */
export const TrafficConfig = {
  /** Default number of driven cars kept alive around the player. */
  density: 25,
  /** Spawn ring around the player (lane points, outside the camera frustum). */
  spawnMin: 90,
  spawnMax: 180,
  /** Cars farther than this from the player are recycled. */
  despawnDistance: 220,
  /** Keep a new car this far from any existing vehicle. */
  spawnClearance: 9,
  spawnInterval: 0.25,
  spawnsPerTick: 2,
  /** Stuck cars are recycled once out of view after this long. */
  stuckDespawnSeconds: 25,

  driver: {
    /** Target speed = edge.speedLimit · random(min, max), clamped to [floor, ceiling]. */
    speedVarianceMin: 0.9,
    speedVarianceMax: 1.12,
    speedFloor: 8,
    speedCeiling: 28,
    /** Speed through an intersection when the next lane turns more than `turnAngle`. */
    turnSpeed: 7,
    turnAngle: 0.5,
    turnSlowDistance: 16,
    /** Lookahead distance along the lane = max(min, speed · factor). */
    lookaheadMin: 6,
    lookaheadFactor: 0.7,
    /** Heading error that saturates the steering command. */
    steerGain: 1.6,
    /** Car-following probe length and the gap held behind a stopped car. */
    probeDistance: 22,
    probeHalfWidth: 2.1,
    stopGap: 4,
    /** Comfortable deceleration used to plan stops (m/s²). */
    planDecel: 4.5,
    /** Distance before the stop line where the car aims to stop. */
    stopLineOffset: 1.2,
    /** A car this close to the line at this speed commits through a red/yellow. */
    commitDistance: 4,
    commitSpeed: 6,
    /** Yellow is treated as red when the car can stop within speed · this + margin. */
    yellowStopFactor: 1.0,
    yellowStopMargin: 3,
    throttleGain: 0.45,
    brakeGain: 0.35,
    /** Blocked timers. */
    honkAfter: 3,
    honkCooldown: 4,
    rerouteAfter: 8,
    /** Pedestrian (player on foot) treated as an obstacle within this corridor half-width. */
    pedHalfWidth: 1.6,
  },

  lights: {
    greenSeconds: 12,
    allRedSeconds: 2,
    /** Last part of the green phase shown as yellow. */
    yellowSeconds: 2.5,
  },

  panic: {
    crimeRadius: 30,
    explosionRadius: 40,
  },
} as const;
