import type { Game } from '../core/Game';
import type { WeaponId } from '../core/peers';
import { WEAPON_ORDER, WEAPONS } from '../data/weapons';
import type { Inventory } from '../systems/weapons/Inventory';
import { WeaponsConfig } from '../systems/weapons/config';
import './weapon-wheel.css';

const RADIUS_PX = 190;
const BLOCKER_ID = 'weaponWheel';

/**
 * Hold Tab: slow-mo radial overlay; mouse movement picks a slot; release equips.
 * Owns its input blocker so the camera and player ignore the mouse while it is open.
 */
export class WeaponWheel {
  private readonly root: HTMLDivElement;
  private readonly slots: HTMLDivElement[] = [];
  private readonly ammoEls: HTMLSpanElement[] = [];
  private readonly currentEl: HTMLSpanElement;
  private open = false;
  private selected: WeaponId = 'fists';
  private selX = 0;
  private selY = 0;

  constructor(
    private readonly game: Game,
    parent: HTMLElement,
    private readonly inventory: Inventory,
    private readonly onEquip: (id: WeaponId) => void,
  ) {
    this.root = document.createElement('div');
    this.root.className = 'weapon-wheel';
    this.root.hidden = true;
    const ring = document.createElement('div');
    ring.className = 'weapon-wheel-ring';
    WEAPON_ORDER.forEach((id, i) => {
      const def = WEAPONS[id];
      const angle = -Math.PI / 2 + (i * Math.PI * 2) / WEAPON_ORDER.length;
      const slot = document.createElement('div');
      slot.className = 'weapon-wheel-slot';
      slot.style.left = `${Math.cos(angle) * RADIUS_PX}px`;
      slot.style.top = `${Math.sin(angle) * RADIUS_PX}px`;
      slot.innerHTML = `<span class="ww-key">${def.slot}</span><span class="ww-name">${def.name}</span><span class="ww-ammo"></span>`;
      ring.append(slot);
      this.slots.push(slot);
      this.ammoEls.push(slot.querySelector('.ww-ammo') as HTMLSpanElement);
    });
    const center = document.createElement('div');
    center.className = 'weapon-wheel-center';
    center.innerHTML = '<span class="ww-current"></span><span class="ww-hint">move to select · release to equip</span>';
    this.currentEl = center.querySelector('.ww-current') as HTMLSpanElement;
    ring.append(center);
    this.root.append(ring);
    parent.append(this.root);
  }

  get isOpen(): boolean {
    return this.open;
  }

  get element(): HTMLElement {
    return this.root;
  }

  /** `canOpen` is evaluated by the weapon system (alive, not modal, on foot or in a car). */
  update(canOpen: boolean): void {
    const input = this.game.input;
    const held = input.isDown('Tab');
    if (!this.open) {
      if (held && canOpen) this.show();
      return;
    }
    if (!held || !this.game.player.isAlive) {
      this.hide();
      return;
    }
    // Accumulate a selection vector; clamp its length so a long drag does not "stick".
    this.selX += input.mouseDeltaX;
    this.selY += input.mouseDeltaY;
    const len = Math.hypot(this.selX, this.selY);
    if (len > 140) {
      this.selX *= 140 / len;
      this.selY *= 140 / len;
    }
    if (len > WeaponsConfig.wheel.selectThresholdPx) {
      const step = (Math.PI * 2) / WEAPON_ORDER.length;
      const a = Math.atan2(this.selY, this.selX) + Math.PI / 2;
      const idx = ((Math.round(a / step) % WEAPON_ORDER.length) + WEAPON_ORDER.length) % WEAPON_ORDER.length;
      const id = WEAPON_ORDER[idx];
      if (id !== this.selected) {
        this.selected = id;
        this.game.audio.play('ui.wheel.tick');
      }
    }
    this.refresh();
  }

  private show(): void {
    this.open = true;
    this.selected = this.inventory.current;
    this.selX = 0;
    this.selY = 0;
    this.game.input.pushBlocker(BLOCKER_ID);
    this.game.time.scale = WeaponsConfig.wheel.timeScale;
    this.root.hidden = false;
    this.refresh();
    this.game.audio.play('ui.wheel.open');
    this.game.events.emit('weapon:wheelToggled', { open: true });
  }

  private hide(): void {
    this.open = false;
    this.game.input.popBlocker(BLOCKER_ID);
    // Only undo our own slow-mo; a pause or card may have changed the scale meanwhile.
    if (this.game.time.scale === WeaponsConfig.wheel.timeScale) this.game.time.scale = 1;
    this.root.hidden = true;
    if (this.selected !== this.inventory.current && this.inventory.has(this.selected)) {
      this.onEquip(this.selected);
      this.game.audio.play('ui.wheel.select');
    }
    this.game.events.emit('weapon:wheelToggled', { open: false });
  }

  private refresh(): void {
    for (let i = 0; i < WEAPON_ORDER.length; i++) {
      const id = WEAPON_ORDER[i];
      const slot = this.slots[i];
      const owned = this.inventory.has(id);
      slot.classList.toggle('owned', owned);
      slot.classList.toggle('locked', !owned);
      slot.classList.toggle('selected', id === this.selected);
      slot.classList.toggle('current', id === this.inventory.current);
      const ammo = this.inventory.slot(id);
      const text = !owned ? 'not owned' : id === 'fists' ? '—' : `${ammo?.mag ?? 0} / ${ammo?.reserve ?? 0}`;
      if (this.ammoEls[i].textContent !== text) this.ammoEls[i].textContent = text;
    }
    const name = WEAPONS[this.selected].name;
    if (this.currentEl.textContent !== name) this.currentEl.textContent = name;
  }

  dispose(): void {
    if (this.open) this.hide();
    this.root.remove();
  }
}
