import type { BlipKind } from '../core/GameState';

export interface BlipStyle {
  color: string;
  /** Glyph drawn on the square; 'label' takes the first letter of the blip label. */
  glyph: string;
  glyphColor: string;
  /** Rendering shape: square with glyph, plain dot, or area circle. */
  shape: 'square' | 'dot' | 'pin' | 'area';
  /** Kept on the minimap rim when off-screen instead of being culled. */
  pinToRim?: boolean;
  legend: string;
}

const DARK = '#0b0b0f';
const LIGHT = '#ffffff';

export const BLIP_STYLES: Record<BlipKind, BlipStyle> = {
  safehouse: { color: '#4ee06a', glyph: '\u2302', glyphColor: DARK, shape: 'square', legend: 'Safehouse' },
  hospital: { color: '#ff4d4d', glyph: '+', glyphColor: LIGHT, shape: 'square', legend: 'Hospital' },
  police: { color: '#3a7bff', glyph: 'P', glyphColor: LIGHT, shape: 'square', legend: 'Police' },
  prison: { color: '#8c8c96', glyph: '#', glyphColor: DARK, shape: 'square', legend: 'Prison' },
  casino: { color: '#ffd23f', glyph: '$', glyphColor: DARK, shape: 'square', legend: 'Casino' },
  gunstore: { color: '#ff8c2a', glyph: 'G', glyphColor: DARK, shape: 'square', legend: 'Gun store' },
  store: { color: '#35e6ff', glyph: '24', glyphColor: DARK, shape: 'square', legend: '24/7' },
  gas: { color: '#ffb347', glyph: 'F', glyphColor: DARK, shape: 'square', legend: 'Gas station' },
  food: { color: '#e9c46a', glyph: 'W', glyphColor: DARK, shape: 'square', legend: 'Food' },
  jewelry: { color: '#f2e6ff', glyph: '\u25C6', glyphColor: '#7a3cff', shape: 'square', legend: 'Jewelry' },
  bank: { color: '#8fd19e', glyph: 'B', glyphColor: DARK, shape: 'square', legend: 'Bank' },
  respray: { color: '#ff4fa3', glyph: 'R', glyphColor: DARK, shape: 'square', legend: 'Respray' },
  chopshop: { color: '#c48a5a', glyph: 'C', glyphColor: DARK, shape: 'square', legend: 'Chop shop' },
  marina: { color: '#4fb8c9', glyph: 'M', glyphColor: DARK, shape: 'square', legend: 'Marina' },
  club: { color: '#a35bff', glyph: '\u266A', glyphColor: LIGHT, shape: 'square', legend: 'Club' },
  mission: { color: '#ffd23f', glyph: 'label', glyphColor: DARK, shape: 'square', pinToRim: true, legend: 'Mission' },
  contact: { color: '#ffb52e', glyph: 'label', glyphColor: DARK, shape: 'square', legend: 'Contact' },
  objective: { color: '#ffd23f', glyph: '', glyphColor: DARK, shape: 'dot', pinToRim: true, legend: 'Objective' },
  waypoint: { color: '#a35bff', glyph: '', glyphColor: LIGHT, shape: 'pin', pinToRim: true, legend: 'Waypoint' },
  vehicle: { color: '#f2f2f2', glyph: '', glyphColor: DARK, shape: 'dot', legend: 'Vehicle' },
  cop: { color: '#3a7bff', glyph: '', glyphColor: DARK, shape: 'dot', legend: 'Police unit' },
  hotScene: { color: 'rgba(255, 59, 59, 0.35)', glyph: '', glyphColor: DARK, shape: 'area', legend: 'Hot scene' },
};

/** Legend order: places first, then dynamic markers. */
export const LEGEND_KINDS: readonly BlipKind[] = [
  'waypoint',
  'mission',
  'objective',
  'safehouse',
  'hospital',
  'police',
  'casino',
  'gunstore',
  'store',
  'gas',
  'food',
  'bank',
  'jewelry',
  'respray',
  'chopshop',
  'marina',
  'club',
  'prison',
];
