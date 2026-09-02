/** Controls table shown in the pause menu (SYSTEM_PROMPT §18). Keys are display labels. */
export interface ControlRow {
  keys: string[];
  action: string;
}

export const CONTROLS_ON_FOOT: readonly ControlRow[] = [
  { keys: ['W', 'A', 'S', 'D'], action: 'Move' },
  { keys: ['Mouse'], action: 'Look' },
  { keys: ['Shift'], action: 'Sprint' },
  { keys: ['Space'], action: 'Jump' },
  { keys: ['RMB'], action: 'Aim' },
  { keys: ['LMB'], action: 'Fire / punch' },
  { keys: ['R'], action: 'Reload' },
  { keys: ['Tab'], action: 'Weapon wheel (hold)' },
  { keys: ['1', '\u2026', '7'], action: 'Quick weapon' },
  { keys: ['Scroll'], action: 'Cycle weapon' },
  { keys: ['F'], action: 'Enter vehicle' },
  { keys: ['E'], action: 'Interact / rob' },
  { keys: ['Q'], action: 'Character wheel' },
  { keys: ['Caps'], action: 'Adrenaline ability' },
  { keys: ['M'], action: 'Map' },
  { keys: ['\u2191'], action: 'Phone' },
  { keys: ['V'], action: 'Camera distance' },
  { keys: ['Esc'], action: 'Pause' },
  { keys: ['P'], action: 'Quick save' },
];

export const CONTROLS_VEHICLE: readonly ControlRow[] = [
  { keys: ['W', 'S'], action: 'Throttle / brake-reverse' },
  { keys: ['A', 'D'], action: 'Steer' },
  { keys: ['Space'], action: 'Handbrake' },
  { keys: ['F'], action: 'Exit vehicle' },
  { keys: ['H'], action: 'Horn' },
  { keys: ['L'], action: 'Lights' },
  { keys: ['C'], action: 'Camera' },
  { keys: ['LMB'], action: 'Fire out the window' },
  { keys: ['Mouse'], action: 'Look around (auto-centers)' },
];
