import {
  DOUBLE_ZERO,
  POCKET_COUNT,
  describeBet,
  isValidBet,
  pocketColor,
  pocketLabel,
  settle,
  spin as spinWheel,
  type EvenMoneyKind,
  type GroupBetKind,
  type InsideBetKind,
  type RouletteBet,
} from '../../casino/Roulette';
import type { CasinoSystem } from '../../systems/casino';
import { CasinoConfig } from '../../systems/casino/config';
import { CasinoPanel } from './Panel';

type OutsideKey = EvenMoneyKind | `dozen${0 | 1 | 2}` | `column${0 | 1 | 2}`;

interface Placement {
  index: number;
  amount: number;
}

/**
 * Clickable American roulette layout. Number cells place straight bets by default; the
 * mode buttons switch to split (two adjacent clicks), street (row of three) or corner.
 */
export class RoulettePanel extends CasinoPanel {
  readonly bets: RouletteBet[] = [];
  private readonly history: Placement[] = [];
  private chip: number = CasinoConfig.roulette.chipValues[0];
  private mode: InsideBetKind = 'straight';
  private pendingSplit: number | null = null;
  private spinning = false;
  private readonly cells = new Map<string, HTMLElement>();
  private chipBtns: HTMLButtonElement[] = [];
  private modeBtns: HTMLButtonElement[] = [];
  private display!: HTMLElement;
  private betsList!: HTMLElement;
  private totalEl!: HTMLElement;
  private spinBtn!: HTMLButtonElement;
  private lastHit: HTMLElement | null = null;
  private raf = 0;

  constructor(casino: CasinoSystem) {
    super(casino, 'roulette', 'Roulette');
  }

  protected build(): void {
    const wrap = this.div('rl-wrap');
    wrap.append(this.buildLayout(), this.buildSide());
    const chipRow = this.row('Chip');
    for (const v of CasinoConfig.roulette.chipValues) {
      const b = this.button(String(v), () => this.selectChip(v), 'secondary', `chip-${v}`);
      this.chipBtns.push(b);
      chipRow.appendChild(b);
    }
    const modeRow = this.row('Inside bet');
    for (const m of ['straight', 'split', 'street', 'corner'] as InsideBetKind[]) {
      const b = this.button(m, () => this.setMode(m), 'secondary small', `mode-${m}`);
      this.modeBtns.push(b);
      modeRow.appendChild(b);
    }
    modeRow.appendChild(this.div('casino-hint', 'Straight 35:1 · Split 17:1 · Street 11:1 · Corner 8:1 · Dozens/Columns 2:1 · Even money 1:1'));
    this.body.append(wrap, chipRow, modeRow, this.msgEl);
    this.selectChip(this.chip);
    this.setMode('straight');
    this.renderBets();
  }

  private buildLayout(): HTMLElement {
    const grid = this.div('rl-layout');
    const cell = (label: string, cls: string, key: string, onClick: () => void, col: number, row: number, colSpan = 1, rowSpan = 1): HTMLElement => {
      const el = this.div(`rl-cell ${cls}`, label);
      el.style.gridColumn = `${col} / span ${colSpan}`;
      el.style.gridRow = `${row} / span ${rowSpan}`;
      el.dataset.cell = key;
      el.addEventListener('click', onClick);
      grid.appendChild(el);
      this.cells.set(key, el);
      return el;
    };
    cell('00', 'green', 'p37', () => this.clickNumber(DOUBLE_ZERO), 1, 1);
    cell('0', 'green', 'p0', () => this.clickNumber(0), 1, 2, 1, 2);
    for (let n = 1; n <= 36; n++) {
      const col = Math.floor((n - 1) / 3) + 2;
      const row = 3 - ((n - 1) % 3);
      cell(String(n), pocketColor(n), `p${n}`, () => this.clickNumber(n), col, row);
    }
    for (let r = 0; r < 3; r++) {
      const index = (2 - r) as 0 | 1 | 2;
      cell('2 TO 1', 'outside', `column${index}`, () => this.placeOutside('column', index), 14, r + 1);
    }
    const dozens = ['1ST 12', '2ND 12', '3RD 12'];
    for (let d = 0; d < 3; d++) {
      const index = d as 0 | 1 | 2;
      cell(dozens[d], 'outside', `dozen${index}`, () => this.placeOutside('dozen', index), 2 + d * 4, 4, 4);
    }
    const even: { label: string; kind: EvenMoneyKind; cls: string }[] = [
      { label: '1 TO 18', kind: 'low', cls: 'outside' },
      { label: 'EVEN', kind: 'even', cls: 'outside' },
      { label: 'RED', kind: 'red', cls: 'red' },
      { label: 'BLACK', kind: 'black', cls: 'black' },
      { label: 'ODD', kind: 'odd', cls: 'outside' },
      { label: '19 TO 36', kind: 'high', cls: 'outside' },
    ];
    even.forEach((e, i) => cell(e.label, e.cls, e.kind, () => this.placeOutside(e.kind), 2 + i * 2, 5, 2));
    return grid;
  }

  private buildSide(): HTMLElement {
    const side = this.div('rl-side');
    this.display = this.div('rl-display', '—');
    this.display.dataset.field = 'outcome';
    this.totalEl = this.div('casino-label', 'On table: 0');
    this.betsList = this.div('rl-bets');
    this.spinBtn = this.button('Spin', () => this.spin(), '', 'spin');
    const undo = this.button('Undo', () => this.undo(), 'secondary small', 'undo');
    const clear = this.button('Clear', () => this.clear(), 'secondary small', 'clear');
    side.append(this.display, this.totalEl, this.betsList, this.row(this.spinBtn, undo, clear));
    return side;
  }

  protected override canClose(): boolean {
    return !this.spinning;
  }

  protected override onClosed(): void {
    cancelAnimationFrame(this.raf);
    // Unspun chips go back to the player.
    for (const b of this.bets) this.casino.giveChips(b.amount);
    this.bets.length = 0;
    this.history.length = 0;
  }

  // --- Betting -----------------------------------------------------------------

  selectChip(value: number): void {
    this.chip = value;
    for (const b of this.chipBtns) b.classList.toggle('active', b.textContent === String(value));
  }

  setMode(mode: InsideBetKind): void {
    this.mode = mode;
    this.pendingSplit = null;
    this.clearSelection();
    for (const b of this.modeBtns) b.classList.toggle('active', b.dataset.action === `mode-${mode}`);
  }

  private clickNumber(pocket: number): void {
    if (this.spinning) return;
    switch (this.mode) {
      case 'straight':
        this.placeInside('straight', [pocket]);
        return;
      case 'split':
        if (this.pendingSplit === null || this.pendingSplit === pocket) {
          this.pendingSplit = pocket;
          this.clearSelection();
          this.cells.get(`p${pocket}`)?.classList.add('selected');
          this.message('Pick the adjacent number for the split.', 'info');
          return;
        }
        this.placeInside('split', [this.pendingSplit, pocket]);
        this.pendingSplit = null;
        this.clearSelection();
        return;
      case 'street': {
        if (pocket === 0 || pocket === DOUBLE_ZERO) return this.fail('Streets are rows of three numbers.');
        const base = pocket - ((pocket - 1) % 3);
        this.placeInside('street', [base, base + 1, base + 2]);
        return;
      }
      case 'corner': {
        if (pocket === 0 || pocket === DOUBLE_ZERO) return this.fail('Corners need four numbers.');
        const anchor = pocket % 3 === 0 ? pocket - 1 : pocket;
        if (anchor > 33) return this.fail('No corner past 36.');
        this.placeInside('corner', [anchor, anchor + 1, anchor + 3, anchor + 4]);
        return;
      }
    }
  }

  placeStraight(pocket: number): void {
    this.placeInside('straight', [pocket]);
  }

  placeInside(kind: InsideBetKind, pockets: number[]): void {
    this.place({ kind, pockets: [...pockets].sort((a, b) => a - b), amount: this.chip });
  }

  placeOutside(kind: EvenMoneyKind | GroupBetKind, index?: 0 | 1 | 2): void {
    if (kind === 'dozen' || kind === 'column') this.place({ kind, index: index ?? 0, amount: this.chip });
    else this.place({ kind, amount: this.chip });
  }

  private place(bet: RouletteBet): void {
    if (this.spinning) return;
    if (!isValidBet(bet)) return this.fail('That is not a valid bet on this layout.');
    const existing = this.bets.findIndex((b) => betKey(b) === betKey(bet));
    if (existing < 0 && this.bets.length >= CasinoConfig.roulette.maxBetsOnTable) return this.fail('Too many bets on the table.');
    if (!this.casino.takeChips(bet.amount)) return this.fail('Not enough chips.');
    let index = existing;
    if (existing >= 0) this.bets[existing].amount += bet.amount;
    else index = this.bets.push(bet) - 1;
    this.history.push({ index, amount: bet.amount });
    this.casino.game.audio.play('casino.chip.place');
    this.message('');
    this.renderBets();
  }

  undo(): void {
    if (this.spinning) return;
    const last = this.history.pop();
    if (!last) return;
    const bet = this.bets[last.index];
    bet.amount -= last.amount;
    this.casino.giveChips(last.amount);
    if (bet.amount <= 0) {
      this.bets.splice(last.index, 1);
      for (const h of this.history) if (h.index > last.index) h.index--;
    }
    this.casino.game.audio.play('casino.chip.collect');
    this.renderBets();
  }

  clear(): void {
    if (this.spinning) return;
    for (const b of this.bets) this.casino.giveChips(b.amount);
    this.bets.length = 0;
    this.history.length = 0;
    this.casino.game.audio.play('casino.chip.collect');
    this.renderBets();
  }

  private clearSelection(): void {
    for (const el of this.cells.values()) el.classList.remove('selected');
  }

  private renderBets(): void {
    for (const el of this.cells.values()) el.querySelector('.rl-chip')?.remove();
    let total = 0;
    this.betsList.replaceChildren();
    for (const b of this.bets) {
      total += b.amount;
      const key = this.cellKeyFor(b);
      const cellEl = this.cells.get(key);
      if (cellEl) {
        const chip = this.div('rl-chip', b.kind === 'straight' || !('pockets' in b) ? String(b.amount) : `${b.kind[0].toUpperCase()}${b.amount}`);
        cellEl.appendChild(chip);
      }
      const line = document.createElement('div');
      line.append(Object.assign(document.createElement('span'), { textContent: describeBet(b) }), Object.assign(document.createElement('span'), { textContent: String(b.amount) }));
      this.betsList.appendChild(line);
    }
    this.totalEl.textContent = `On table: ${total.toLocaleString('en-US')}`;
    this.spinBtn.disabled = this.spinning || this.bets.length === 0;
  }

  private cellKeyFor(bet: RouletteBet): string {
    if ('pockets' in bet) return `p${bet.pockets[0]}`;
    if (bet.kind === 'dozen' || bet.kind === 'column') return `${bet.kind}${bet.index}` as OutsideKey;
    return bet.kind;
  }

  // --- Spin -------------------------------------------------------------------

  spin(): void {
    if (this.spinning) return;
    if (this.bets.length === 0) return this.fail('Place a bet first.');
    const rigged = this.casino.rig.takeRoulette();
    const outcome = rigged ?? spinWheel(this.casino.rng);
    this.spinning = true;
    this.pendingSplit = null;
    this.clearSelection();
    this.lastHit?.classList.remove('hit');
    this.spinBtn.disabled = true;
    this.message('No more bets.', 'info');
    const seconds = CasinoConfig.roulette.spinSeconds;
    const audio = this.casino.game.audio;
    const loop = audio.loop('casino.roulette.spin.loop');
    const ball = audio.loop('casino.roulette.ball.loop');
    this.casino.stage.spinRoulette(outcome, seconds);
    const start = performance.now();
    let nextFlip = 0;
    const tick = (): void => {
      const t = Math.min(1, (performance.now() - start) / (seconds * 1000));
      if (t >= 1) {
        this.showOutcome(outcome);
        audio.stop(loop);
        audio.stop(ball);
        this.resolve(outcome);
        return;
      }
      // Rolling number strip: flips slow down as the ball loses speed.
      if (performance.now() >= nextFlip) {
        const random = Math.min(POCKET_COUNT - 1, Math.floor(this.casino.rng() * POCKET_COUNT));
        this.showOutcome(random);
        nextFlip = performance.now() + 40 + t * t * 600;
      }
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  private showOutcome(pocket: number): void {
    this.display.textContent = pocketLabel(pocket);
    this.display.className = `rl-display ${pocketColor(pocket)}`;
  }

  private resolve(outcome: number): void {
    const s = settle(this.bets, outcome);
    this.casino.game.audio.play('casino.roulette.drop');
    this.casino.giveChips(s.returned);
    this.casino.settleRound('roulette', s.net);
    const hit = this.cells.get(`p${outcome}`);
    if (hit) {
      hit.classList.add('hit');
      this.lastHit = hit;
    }
    const label = `${pocketLabel(outcome)} ${pocketColor(outcome).toUpperCase()}`;
    if (s.net > 0) {
      this.message(`${label} — you win ${s.net.toLocaleString('en-US')} chips!`, 'win');
      this.casino.game.audio.play('casino.roulette.win');
      this.casino.game.audio.play('casino.chip.collect');
    } else if (s.net === 0 && s.returned > 0) {
      this.message(`${label} — you break even.`, 'info');
      this.casino.game.audio.play('casino.push');
    } else {
      this.message(`${label} — house takes ${(-s.net).toLocaleString('en-US')} chips.`, 'lose');
      this.casino.game.audio.play('casino.lose');
    }
    this.bets.length = 0;
    this.history.length = 0;
    this.spinning = false;
    this.renderBets();
  }
}

function betKey(bet: RouletteBet): string {
  if ('pockets' in bet) return `${bet.kind}:${bet.pockets.join('-')}`;
  if (bet.kind === 'dozen' || bet.kind === 'column') return `${bet.kind}:${bet.index}`;
  return bet.kind;
}
