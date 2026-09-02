/**
 * All foundation tunables. Units: meters, seconds, kg, radians (degrees only in UI).
 * Other agents keep their tunables in `src/systems/<id>/config.ts`.
 */
export const Config = {
  seed: 'leonida-2026',

  render: {
    maxPixelRatio: 2,
    fov: 60,
    near: 0.3,
    far: 3000,
    shadows: true,
    shadowMapSize: 2048,
    /** Half-extent of the shadow camera box that follows the player. */
    shadowHalfExtent: 60,
    toneMappingExposure: 1.25,
  },

  loop: {
    fixedDt: 1 / 60,
    maxSubsteps: 3,
    maxFrameDt: 0.1,
  },

  time: {
    /** 24-minute day: 1 real second = 1 game minute. */
    dayLengthSeconds: 24 * 60,
    startHour: 18.5,
    startDay: 1,
  },

  physics: {
    gravity: -9.81,
    characterOffset: 0.05,
  },

  world: {
    size: 2000,
    chunkSize: 200,
    /** Land rectangles (x0, z0, x1, z1). Everything else is water. */
    mainland: { x0: -800, z0: -900, x1: 350, z1: 900 },
    island: { x0: 500, z0: -650, x1: 940, z1: 650 },
    swamp: { x0: -1000, z0: -900, x1: -800, z1: 900 },
    waterLevel: -0.35,
    seabedDepth: -2.6,
    sidewalkWidth: 4,
    sidewalkHeight: 0.15,
    roadSurfaceY: 0.02,
    lotSurfaceY: 0.04,
    floorHeight: 3.2,
    bridgeDeckHeight: 8,
    bridgeRampLength: 60,
    laneWidth: 3.5,
  },

  dayNight: {
    sunriseHour: 6.25,
    sunsetHour: 19.0,
    /** Sun elevation (sin) below which lamps/neon/windows are fully on. */
    lampsOnElevation: 0.02,
    /** Sun elevation above which they are fully off; ramps between. */
    lampsOffElevation: 0.22,
    fogNear: 260,
    fogFar: 1700,
    sunIntensityDay: 3.2,
    moonIntensity: 0.25,
    hemiIntensityDay: 1.5,
    hemiIntensityNight: 0.4,
  },

  player: {
    capsuleRadius: 0.35,
    capsuleHeight: 1.8,
    walkSpeed: 2.0,
    runSpeed: 4.5,
    sprintSpeed: 7.0,
    acceleration: 20,
    deceleration: 28,
    turnRateRadPerSec: (720 * Math.PI) / 180,
    jumpSpeed: 5,
    gravity: -18,
    terminalFall: -40,
    autostepHeight: 0.5,
    autostepMinWidth: 0.2,
    snapToGround: 0.5,
    mass: 80,
    maxHealth: 100,
    maxArmor: 100,
    armorAbsorb: 0.5,
    regenTarget: 50,
    regenDelay: 5,
    regenRate: 5,
    deathDuration: 4,
    deathCashFraction: 0.1,
    deathCashMax: 5000,
    /** Prison gate outside Leonida Penitentiary (mission 1 start). */
    spawn: { x: -200, y: 0, z: -822, heading: Math.PI },
  },

  camera: {
    thirdPerson: { distance: 4.5, height: 1.6, stiffness: 8, lookStiffness: 14, fov: 60 },
    aim: { distance: 1.8, offsetX: 0.6, height: 1.55, fovStart: 50, fovEnd: 40, stiffness: 18 },
    vehicle: {
      minDistance: 6,
      maxDistance: 9,
      height: 2.5,
      stiffness: 6,
      fovLow: 60,
      fovHigh: 75,
      fovSpeedMs: 40,
      autoCenterDelay: 1.5,
      autoCenterRate: 3,
    },
    map: { height: 320 },
    pitchMin: -0.55,
    pitchMax: 1.2,
    mouseSensitivity: 0.0022,
    collisionRadius: 0.3,
    pivotHeight: 1.3,
    shakeFrequency: 28,
  },

  economy: {
    startCash: 250,
  },

  interiors: {
    doorRadius: 1.6,
    fadeSeconds: 0.35,
    origin: { x: 0, y: 600, z: 0 },
  },
};

export type ConfigType = typeof Config;
