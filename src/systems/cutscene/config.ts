/** Cutscene tunables (seconds unless noted). */
export const CutsceneConfig = {
  /** Key that skips the whole scene. */
  skipKey: 'Space',
  /** Ignore the skip key for this long after a scene starts so the E/Space that began it does not skip it. */
  skipGrace: 0.4,
  /** Default camera glide when a step omits `seconds`. */
  defaultMoveSeconds: 2.5,
  /** Default orbit when a step omits values. */
  orbit: { radius: 9, height: 3.2, speed: 0.25, fov: 45 },
  inputBlocker: 'cutscene',
} as const;
