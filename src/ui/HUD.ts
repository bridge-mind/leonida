import type { Game } from '../core/Game';
import type { GameState, WantedState } from '../core/GameState';
import { UiConfig } from '../systems/ui/config';
import { clamp01, el, setText, setVisible, toggleClass, type UiComponent } from './dom';
import { formatCash, formatCashDelta, weaponLabel } from './format';
import './hud.css';

const H = UiConfig.hud;
const STAR_SLOTS = 6;

/**
 * Top-right cluster: cash (ticking), wanted stars, weapon/ammo, clock, casino chips.
 * Every DOM write is guarded by a cached last value; nothing is rebuilt per frame.
 */
export class HUD implements UiComponent {
  readonly el: HTMLElement;
  private readonly cashEl: HTMLElement;
  private readonly deltaEl: HTMLElement;
  private readonly chipsEl: HTMLElement;
  private readonly stars: HTMLElement[] = [];
  private readonly weaponRow: HTMLElement;
  private readonly weaponName: HTMLElement;
  private readonly ammoEl: HTMLElement;
  private readonly ammoMag: HTMLElement;
  private readonly ammoReserve: HTMLElement;
  private readonly clockEl: HTMLElement;
  private readonly abilityRow: HTMLElement | null;
  private readonly abilityFill: HTMLElement | null;

  private shownCash = 0;
  private cashInitialised = false;
  private lastCashText = '';
  private deltaTimer = 0;
  private deltaAccum = 0;
  private lastStarLevel = 0;
  private lastStarState: WantedState = 'none';
  private flashTimer = 0;
  private flashLevel = 0;
  private lastChips = -1;
  private lastClock = '';
  private lastWeaponKey = '';
  private lastAbilityPct = -1;
  private readonly offs: (() => void)[] = [];

  constructor(private readonly game: Game) {
    this.el = el('div', 'hud');
    const cashRow = el('div', 'hud-cash-row');
    this.cashEl = el('div', 'hud-cash', '$0');
    this.deltaEl = el('div', 'hud-cash-delta');
    cashRow.append(this.cashEl, this.deltaEl);
    this.chipsEl = el('div', 'hud-chips');
    this.chipsEl.hidden = true;
    const starsRow = el('div', 'hud-stars');
    for (let i = 0; i < STAR_SLOTS; i++) {
      const s = el('span', 'hud-star', '\u2605');
      this.stars.push(s);
      starsRow.appendChild(s);
    }
    this.weaponRow = el('div', 'hud-weapon');
    this.weaponRow.hidden = true;
    this.weaponName = el('span', 'hud-weapon-name');
    this.ammoEl = el('span', 'hud-ammo');
    this.ammoMag = el('span', 'mag');
    this.ammoReserve = el('span', 'reserve');
    this.ammoEl.append(this.ammoMag, this.ammoReserve);
    this.weaponRow.append(this.weaponName, this.ammoEl);
    this.clockEl = el('div', 'hud-clock');
    this.el.append(cashRow, this.chipsEl, starsRow, this.weaponRow, this.clockEl);
    // Special-ability meter only exists when the characters system publishes `state.abilityCharge`.
    if (game.systems.has('characters')) {
      this.abilityRow = el('div', 'hud-ability');
      this.abilityFill = el('div', 'hud-ability-fill');
      const bar = el('div', 'hud-ability-bar');
      bar.appendChild(this.abilityFill);
      this.abilityRow.append(el('span', 'hud-ability-label', 'Ability'), bar);
      this.el.appendChild(this.abilityRow);
    } else {
      this.abilityRow = null;
      this.abilityFill = null;
    }

    this.offs.push(
      game.events.on('economy:cashChanged', (e) => this.onCashChanged(e.delta)),
      game.events.on('wanted:cleared', () => this.startFlash()),
    );
  }

  private onCashChanged(delta: number): void {
    if (delta === 0) return;
    // Rapid same-sign changes (robbery ticks, casino payouts) merge into one floating number.
    if (this.deltaTimer > 0 && Math.sign(this.deltaAccum) === Math.sign(delta)) this.deltaAccum += delta;
    else this.deltaAccum = delta;
    this.deltaTimer = H.cashDeltaSeconds;
    setText(this.deltaEl, formatCashDelta(this.deltaAccum));
    toggleClass(this.deltaEl, 'gain', this.deltaAccum > 0);
    toggleClass(this.deltaEl, 'loss', this.deltaAccum < 0);
  }

  private startFlash(): void {
    this.flashLevel = Math.max(1, this.lastStarLevel);
    this.flashTimer = H.starFlashSeconds;
    for (let i = 0; i < this.flashLevel; i++) {
      this.stars[i].className = 'hud-star searching flash';
    }
  }

  update(state: GameState, realDt: number): void {
    this.updateCash(state, realDt);
    this.updateStars(state, realDt);
    this.updateWeapon(state);
    this.updateChips(state);
    this.updateAbility(state);
    const clock = this.game.time.clockLabel;
    if (clock !== this.lastClock) {
      this.lastClock = clock;
      setText(this.clockEl, clock);
    }
  }

  private updateCash(state: GameState, dt: number): void {
    const target = state.cash;
    if (!this.cashInitialised) {
      this.cashInitialised = true;
      this.shownCash = target;
    } else if (this.shownCash !== target) {
      const diff = target - this.shownCash;
      const rate = Math.max(H.cashTickMinRate, Math.abs(diff) / H.cashTickSeconds);
      const step = rate * dt;
      this.shownCash = Math.abs(diff) <= step ? target : this.shownCash + Math.sign(diff) * step;
    }
    const text = formatCash(Math.round(this.shownCash));
    if (text !== this.lastCashText) {
      this.lastCashText = text;
      setText(this.cashEl, text);
    }
    if (this.deltaTimer > 0) {
      this.deltaTimer -= dt;
      const t = Math.max(0, this.deltaTimer / H.cashDeltaSeconds);
      // Ease out: fully visible for the first half, then fade and drift down.
      const alpha = Math.min(1, t * 2);
      this.deltaEl.style.opacity = alpha.toFixed(2);
      this.deltaEl.style.transform = `translateY(${((1 - t) * 10).toFixed(1)}px)`;
    } else if (this.deltaEl.style.opacity !== '0') {
      this.deltaEl.style.opacity = '0';
      this.deltaAccum = 0;
    }
  }

  private updateStars(state: GameState, dt: number): void {
    if (this.flashTimer > 0) {
      this.flashTimer -= dt;
      if (this.flashTimer > 0) return;
      // Force a class rebuild once the flash ends.
      this.lastStarLevel = -1;
    }
    const level = Math.max(0, Math.min(STAR_SLOTS, Math.round(state.wanted.level)));
    const wantedState = state.wanted.state;
    if (level === this.lastStarLevel && wantedState === this.lastStarState) return;
    this.lastStarLevel = level;
    this.lastStarState = wantedState;
    const cls = level === 0 || wantedState === 'none' ? 'active' : wantedState;
    for (let i = 0; i < STAR_SLOTS; i++) {
      this.stars[i].className = i < level ? `hud-star ${cls}` : 'hud-star';
    }
  }

  private updateWeapon(state: GameState): void {
    const w = state.weapon;
    if (!w || w.id === 'fists') {
      if (this.lastWeaponKey !== '') {
        this.lastWeaponKey = '';
        setVisible(this.weaponRow, false);
      }
      return;
    }
    const key = `${w.id}|${w.ammoInMag}|${w.ammoReserve}|${w.isReloading ? 1 : 0}`;
    if (key === this.lastWeaponKey) return;
    this.lastWeaponKey = key;
    setVisible(this.weaponRow, true);
    setText(this.weaponName, weaponLabel(w.id));
    setText(this.ammoMag, `${w.ammoInMag}`);
    setText(this.ammoReserve, ` / ${w.ammoReserve}`);
    toggleClass(this.ammoEl, 'reloading', w.isReloading);
  }

  private updateChips(state: GameState): void {
    const show = state.interior === 'casino';
    const chips = show ? Math.round(state.chips) : -1;
    if (chips === this.lastChips) return;
    this.lastChips = chips;
    setVisible(this.chipsEl, show);
    if (show) setText(this.chipsEl, `${chips.toLocaleString('en-US')} CHIPS`);
  }

  private updateAbility(state: GameState): void {
    if (!this.abilityRow || !this.abilityFill) return;
    const pct = Math.round(clamp01(state.abilityCharge) * 100);
    if (pct === this.lastAbilityPct) return;
    this.lastAbilityPct = pct;
    this.abilityFill.style.transform = `scaleX(${pct / 100})`;
    toggleClass(this.abilityRow, 'full', pct >= 100);
  }

  dispose(): void {
    for (const off of this.offs) off();
    this.el.remove();
  }
}
