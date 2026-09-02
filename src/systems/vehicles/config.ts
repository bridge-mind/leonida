/** Tunables for the vehicle system. Units: meters, seconds, kg, radians. */
export const VehicleConfig = {
  /** Renderer slot capacity: parked (≤ 250) + traffic + spawned. */
  maxVehicles: 320,
  /** Vehicles beyond this distance from the camera are hidden (scale 0). */
  renderDistance: 260,
  /** Engine/skid loops only for vehicles within this range of the player. */
  audioDistance: 60,

  parked: {
    /** Hard cap on parked cars spawned at init. */
    cap: 250,
    /** Minimum spacing between parked cars (thins the 6.5 m curb spots to ~1 per 40 m of street). */
    minSpacing: 18,
    /** Chance a curb spot is used at all (before spacing). */
    curbChance: 0.6,
    lotChance: 0.45,
  },

  interaction: {
    enterRadius: 3,
    /** Player walks toward the door for this long before mounting. */
    enterSeconds: 0.8,
    carjackSeconds: 1.2,
    /** Occupied cars faster than this cannot be carjacked; nothing faster is enterable. */
    maxEnterSpeed: 5,
    /** Bailing out above this speed hurts. */
    exitDamageSpeed: 8,
    exitDamagePerMs: 1.5,
    /** Cancel a pending enter if the player wanders farther than this from the car. */
    cancelDistance: 5,
    /** Door offset outward from the body side. */
    doorOutset: 0.9,
  },

  physics: {
    linearDamping: 0.05,
    angularDamping: 2,
    friction: 0.5,
    /** Fraction of the roof height used for the collider so kerbs/ramps slide instead of catching. */
    colliderHeightScale: 0.9,
    /** Extra reach of the grounded raycast below the collider bottom. */
    groundProbe: 0.45,
  },

  handling: {
    /** Speed at which the steering has full authority (crawling cars barely turn). */
    minSteerSpeed: 4,
    /** Steering authority halves at this speed (stability at highway speed). */
    steerFalloffSpeed: 25,
    handbrakeSteerBoost: 1.35,
    /** Handbrake longitudinal deceleration (rear wheels locked). */
    handbrakeDecel: 6,
    /** Yaw rate response (1/s): higher = snappier steering. */
    yawResponse: 9,
    /** Aerodynamic drag: decel = drag · v² (m⁻¹). */
    drag: 0.0004,
    /** Rolling resistance (m/s²). */
    rolling: 0.4,
    /** Near-zero speed below which brake input starts reversing. */
    brakeToReverseSpeed: 0.5,
    /** Engine accel scales by (1 − k · ratio²) toward top speed. */
    topSpeedFalloff: 0.7,
    /** Visual front wheel steer angle at full lock. */
    wheelSteerAngle: 0.5,
    /** Lateral speed above which tires screech (skid loop / tire smoke). */
    skidLateralSpeed: 4,
  },

  damage: {
    /** Collision Δv below this is a parking bump: no damage, no event. */
    minDeltaV: 1.5,
    /** Damage per m/s of Δv beyond the threshold, scaled by (mass / 1200). */
    damagePerDeltaV: 12,
    /** Δv thresholds for the three crash sounds. */
    crashMedium: 4,
    crashHeavy: 10,
    smokeHealth: 300,
    smokeInterval: 0.15,
    fireInterval: 0.1,
    /** Seconds on fire before the explosion. */
    fireSeconds: 6,
    /** Damage per second to occupants of a burning car. */
    fireDps: 5,
    explosionRadius: 6,
    explosionDamage: 100,
    explosionImpulse: 8000,
    /** Bullets hitting the rear third (gas tank) do double damage. */
    gasTankMultiplier: 2,
    /** Wrecks despawn after this long once far enough from the player. */
    wreckDespawnSeconds: 30,
    wreckDespawnDistance: 60,
    /** Body colour multipliers by health tier (≥ 700, ≥ 300, > 0, wrecked). */
    tierDarkening: [1, 0.8, 0.55, 0.12] as readonly number[],
    cameraShakePerDeltaV: 0.02,
    maxCameraShake: 0.5,
  },

  lights: {
    /** Headlights come on automatically above this night level (0..1). */
    autoOnNight: 0.5,
    /** Police light bar red/blue alternation period. */
    sirenFlashPeriod: 0.5,
  },

  audio: {
    /** Engine "gears" used to shape rpm from speed. */
    gears: 5,
    idleRpm: 0.18,
  },
} as const;
