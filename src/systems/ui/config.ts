/** UI tunables. Units: seconds, CSS pixels, meters (px/m for map zoom). */
export const UiConfig = {
  hud: {
    /** Cash counter ticks at least this many dollars per second, or the whole delta in one second. */
    cashTickMinRate: 20,
    cashTickSeconds: 1,
    cashDeltaSeconds: 2,
    starFlashSeconds: 2,
  },
  minimap: {
    width: 260,
    height: 200,
    cornerRadius: 6,
    /** Zoom on foot and at top speed (px per meter); blends by vehicle speed. */
    zoomOnFoot: 1.3,
    zoomAtSpeed: 0.6,
    zoomSpeedKmh: 120,
    zoomSmoothing: 3,
    blipSize: 11,
    dotSize: 5,
    routeWidth: 3,
    /** Waypoint and mission blips stick to the rim when off-screen, this far inside the edge. */
    rimMargin: 9,
    /** How far around the player cops/vehicles are queried for dots. */
    unitQueryRadius: 260,
    maxUnits: 64,
  },
  map: {
    /** Offscreen static layer resolution. 1 px/m keeps roads crisp when the full map zooms in. */
    staticPxPerMeter: 1,
    halfExtent: 1000,
    minZoom: 0.3,
    maxZoom: 4,
    startZoom: 1.0,
    wheelZoomFactor: 1.18,
    blipSize: 14,
    routeWidth: 4,
    /** Clicking within this many meters of the current waypoint clears it. */
    clearRadius: 20,
    /** Mouse movement below this many pixels between down and up still counts as a click. */
    clickSlop: 4,
    colors: {
      water: '#16324f',
      land: '#a9a394',
      roadMinor: '#d8d3c6',
      roadMajor: '#efebe0',
      roadHighway: '#fff8e8',
      roadDirt: '#b39a72',
      roadOutline: '#5f5b52',
      footprint: 'rgba(40, 40, 44, 0.28)',
      route: '#a35bff',
    },
  },
  notifications: {
    seconds: 4,
    max: 3,
    /** Cash changes below this are shown on the HUD delta only, not as a feed entry. */
    moneyThreshold: 100,
  },
  cards: {
    defaultSeconds: 4,
    fadeSeconds: 0.6,
    slowMoScale: 0.25,
  },
  dialogue: {
    secondsPerChar: 0.06,
    minSeconds: 1.8,
    maxSeconds: 6,
  },
  pause: {
    statsRefreshSeconds: 0.5,
  },
  title: {
    orbitCenter: { x: 700, y: 0, z: 0 },
    orbitRadius: 260,
    orbitHeight: 70,
    orbitSpeed: 0.04,
  },
  settings: {
    storageKey: 'leonida.settings',
    sensitivityMin: 0.0008,
    sensitivityMax: 0.005,
  },
} as const;
