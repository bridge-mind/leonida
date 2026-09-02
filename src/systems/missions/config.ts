/** Mission system tunables. Units: meters, seconds, dollars. */
export const MissionsConfig = {
  marker: {
    /** Walking this close to a contact marker shows the E prompt. */
    radius: 3,
    height: 2.4,
    color: 0xffd23f,
    opacity: 0.35,
    /** Bob amplitude of the marker glow (visual only). */
    pulseSeconds: 2.2,
  },
  /** Objective "target lost" distance for chase beats. */
  targetLostDistance: 400,
  escape: {
    /** escapeWanted with no stars yet: seconds outdoors at level 0 before it counts as clean. */
    graceSeconds: 3,
  },
  cards: {
    seconds: 4,
    /** Card duration while the debug `complete` action drives a mission. */
    fastSeconds: 0.2,
    /** Seconds after respawn before the MISSION FAILED card (lets WASTED finish). */
    afterRespawnDelay: 0.6,
  },
  casino: {
    /** Cash below this with zero chips counts as broke. */
    brokeCash: 100,
    brokeTaunt: 'Dominic: "A tourist after all. Get out of my casino."',
  },
  /** Autostart the first mission this long after New Game (seconds). */
  newGameStartDelay: 1.5,
  /** Waypoint refresh: objectives whose target moved more than this re-route (meters). */
  waypointMoveThreshold: 4,
} as const;
