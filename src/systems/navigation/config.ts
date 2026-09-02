/** Navigation tunables. Units: meters, seconds. */
export const NavConfig = {
  /** Player farther than this from the nearest route segment triggers a re-route. */
  strayDistance: 30,
  /** How often the stray check runs (per frame would be wasteful for a polyline walk). */
  strayCheckInterval: 0.5,
  /** Waypoint auto-clears when the player is this close. */
  arriveDistance: 12,
  /** Route points closer than this to the target are dropped so the line ends at the pin, not past it. */
  finalPointMergeDistance: 8,
  /** Route line sits slightly above the road for anything that renders it in 3D later. */
  routeY: 0.3,
} as const;
