/** Weapon system tunables. Units: meters, seconds, radians. */
export const WeaponsConfig = {
  /** Debug convenience: spawn with a pistol and two spare mags. */
  startWithPistol: false,
  /** Picking up a gun while unarmed equips it. */
  autoEquipOnGive: true,
  /** Start a reload automatically when the magazine runs dry and reserve remains. */
  autoReload: true,
  /** Hip fire (not aiming) widens the cone by this factor. */
  hipFireSpreadMul: 2,
  /**
   * With a scripted aim point (autotests) the first projectile flies exactly at it so
   * damage assertions are deterministic; extra pellets still spread.
   */
  exactScriptedAim: true,

  aimAssist: true,
  aimAssistConeRad: (10 * Math.PI) / 180,
  aimAssistRange: 40,
  /** Height above the feet the aim-assist snaps to. */
  aimAssistChestHeight: 1.2,
  /** Range for `weapon:aimedAt` target detection. */
  aimTargetRange: 40,

  /** Sniper scope narrows the aim FOV by this many degrees while aiming. */
  sniperScopeFovDelta: -25,

  recoil: { fovKick: 1.2, shake: 0.03, shakeSeconds: 0.08 },

  muzzle: { up: 1.4, right: 0.3, forward: 0.5, vehicleUp: 1.2, vehicleRight: 0.8 },

  melee: {
    damage: 15,
    reach: 1.5,
    /** Seconds between hits inside a combo. */
    hitInterval: 0.35,
    /** Idle seconds after which the combo resets to the first hit. */
    comboReset: 1,
    knockdownForward: 180,
    knockdownUp: 60,
    /** Peds within this radius are candidates; a facing test picks one in front. */
    queryRadius: 2,
    facingDot: 0.5,
  },

  npc: {
    /** Spread multiplier at accuracy 0: spread × (1 + spreadPenalty × (1 − accuracy)). */
    spreadPenalty: 3,
    /** Chance of a deliberate wide miss at accuracy 0. */
    missChance: 0.5,
    /** Radians added on a deliberate miss. */
    missOffsetRad: 0.12,
  },

  dummy: { width: 0.8, height: 1.8, depth: 0.5, headHeight: 1.5, health: 100 },

  wheel: { timeScale: 0.2, selectThresholdPx: 28 },
};
