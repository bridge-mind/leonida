/** Text formatting shared by HUD, notifications and menus. */

export function formatCash(amount: number): string {
  const n = Math.round(Math.abs(amount));
  return `${amount < 0 ? '-' : ''}$${n.toLocaleString('en-US')}`;
}

export function formatCashDelta(delta: number): string {
  return `${delta >= 0 ? '+' : '-'}$${Math.round(Math.abs(delta)).toLocaleString('en-US')}`;
}

/** `350 m` under a kilometre, `1.2 km` above. */
export function formatDistance(meters: number): string {
  if (meters < 1000) return `${Math.round(meters / 5) * 5} m`;
  return `${(meters / 1000).toFixed(1)} km`;
}

export function formatDuration(seconds: number): string {
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  const mm = m.toString().padStart(2, '0');
  const ss = r.toString().padStart(2, '0');
  return h > 0 ? `${h}:${mm}:${ss}` : `${m}:${ss}`;
}

const REASON_LABELS: Record<string, string> = {
  hospital: 'Hospital bill',
  bail: 'Bail',
  pickup: 'Cash picked up',
  chopshop: 'Vehicle sold',
  chop_shop: 'Vehicle sold',
  robbery: 'Store robbed',
  rob_store: 'Store robbed',
  rob_ped: 'Wallet taken',
  mission: 'Mission reward',
  casino: 'Casino',
  chips: 'Casino chips',
  shop: 'Purchase',
  gunstore: 'Bullet Barn',
  ammo: 'Ammunition',
  food: 'Food',
  respray: 'Respray',
  debug: 'Debug',
  test: 'Test',
};

/** Turns an economy reason id into feed text; unknown ids are capitalised with underscores spaced. */
export function humaniseReason(reason: string): string {
  const known = REASON_LABELS[reason];
  if (known) return known;
  const spaced = reason.replace(/[_-]+/g, ' ').trim();
  return spaced.length === 0 ? 'Cash' : spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

const WEAPON_LABELS: Record<string, string> = {
  pistol: 'Pistol',
  smg: 'SMG',
  shotgun: 'Shotgun',
  rifle: 'Assault Rifle',
  sniper: 'Sniper Rifle',
};

export function weaponLabel(id: string): string {
  return WEAPON_LABELS[id] ?? id.charAt(0).toUpperCase() + id.slice(1);
}

const MARKUP_COLORS: Record<string, string> = { y: 'yellow', b: 'blue', g: 'green', r: 'red' };

/**
 * Renders GTA-style `~y~word~y~` markup into `target` as colored spans. The text is
 * inserted with `textContent`, never `innerHTML`, so objective strings are safe.
 */
export function renderMarkup(target: HTMLElement, text: string): void {
  target.replaceChildren();
  const re = /~([ybgr])~/g;
  let last = 0;
  let color: string | null = null;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    const chunk = text.slice(last, m.index);
    if (chunk) target.appendChild(span(chunk, color));
    color = color === null ? MARKUP_COLORS[m[1]] : null;
    last = m.index + m[0].length;
  }
  const tail = text.slice(last);
  if (tail) target.appendChild(span(tail, color));
}

function span(text: string, color: string | null): Node {
  if (!color) return document.createTextNode(text);
  const s = document.createElement('span');
  s.className = `markup-${color}`;
  s.textContent = text;
  return s;
}

/** Stable hue from a name so a speaker keeps their color between lines. */
export function nameColor(name: string): string {
  const key = name.trim().toLowerCase();
  if (key === 'lucia') return 'var(--color-accent-pink)';
  if (key === 'jason') return 'var(--color-accent-cyan)';
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) >>> 0;
  return `hsl(${h % 360} 85% 68%)`;
}
