export const ShopsConfig = {
  clerk: {
    /** Seconds the player may keep a gun drawn (not aimed at the clerk) before an armed clerk reacts. */
    drawnGraceSeconds: 2,
    /** During a hold-up: seconds before an armed clerk tries for the shotgun when not covered. */
    robberyGraceSeconds: 3,
    /** Gap in aim contact that counts as "not covered". */
    aimGapSeconds: 0.5,
    fireInterval: 1.1,
    accuracy: 0.45,
    /** Distance at which the clerk turns to follow the player. */
    attentionRadius: 9,
  },
  guard: {
    /** Seconds between spotting a drawn weapon and opening fire (the "drop it!" window). */
    warningSeconds: 0.8,
    fireInterval: 0.75,
    accuracy: 0.5,
    attentionRadius: 14,
  },
  /** Muzzle / chest heights used for NPC fire origins and targets. */
  chestHeight: 1.35,
  playerChestHeight: 1.2,
  /** Player must be within this many metres of the counter point to get the prompt. */
  counterRadius: 1.8,
  /** In-game days a robbed store stays shut. */
  closedDays: 1,
} as const;

export const ResprayConfig = {
  /** Radius of the drive-in volume in front of the garage door. */
  radius: 5.5,
  /** How far in front of the door the volume centre sits. */
  forward: 4.5,
  /** Vehicle must be slower than this (m/s) to get the prompt. */
  maxSpeed: 1,
  holdSeconds: 1,
} as const;
