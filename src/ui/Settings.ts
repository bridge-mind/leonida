import type { Game } from '../core/Game';
import { UiConfig } from '../systems/ui/config';
import { applySettings, loadSettings, saveSettings, type Settings as SettingsData } from '../systems/ui/settings';
import { el } from './dom';
import './menu.css';

type NumericKey = 'mouseSensitivity' | 'masterVolume' | 'sfxVolume' | 'musicVolume';
type BoolKey = 'aimAssist' | 'shadows';

/**
 * Settings form for the pause menu. Every change applies immediately and is persisted
 * to `localStorage['leonida.settings']` so other systems can read it.
 */
export class SettingsPanel {
  readonly el: HTMLElement;
  readonly data: SettingsData;

  constructor(private readonly game: Game) {
    this.data = loadSettings();
    applySettings(game, this.data);
    this.el = el('div', 'settings');
    const { sensitivityMin, sensitivityMax } = UiConfig.settings;
    this.el.append(
      this.slider('Mouse sensitivity', 'mouseSensitivity', sensitivityMin, sensitivityMax, (v) => `${Math.round((v / sensitivityMax) * 100)}`),
      this.slider('Master volume', 'masterVolume', 0, 1, percent),
      this.slider('Effects volume', 'sfxVolume', 0, 1, percent),
      this.slider('Music volume', 'musicVolume', 0, 1, percent),
      this.toggle('Aim assist', 'aimAssist'),
      this.toggle('Shadows', 'shadows'),
      el('div', 'settings-note', 'Settings apply immediately and persist between sessions.'),
    );
  }

  private commit(): void {
    saveSettings(this.data);
    applySettings(this.game, this.data);
  }

  private slider(label: string, key: NumericKey, min: number, max: number, fmt: (v: number) => string): HTMLElement {
    const row = el('div', 'settings-row');
    const input = el('input');
    input.type = 'range';
    input.min = String(min);
    input.max = String(max);
    input.step = String((max - min) / 100);
    input.value = String(this.data[key]);
    const value = el('span', 'settings-value', fmt(this.data[key]));
    input.addEventListener('input', () => {
      this.data[key] = Number(input.value);
      value.textContent = fmt(this.data[key]);
      this.commit();
      this.game.audio.play('ui.blip', { volume: 0.4 });
    });
    row.append(el('span', undefined, label), input, value);
    return row;
  }

  private toggle(label: string, key: BoolKey): HTMLElement {
    const row = el('div', 'settings-row');
    const button = el('button', 'settings-toggle');
    const render = (): void => {
      button.textContent = this.data[key] ? 'ON' : 'OFF';
      button.classList.toggle('on', this.data[key]);
    };
    render();
    button.addEventListener('click', () => {
      this.data[key] = !this.data[key];
      render();
      this.commit();
      this.game.audio.play('ui.select');
    });
    row.append(el('span', undefined, label), button);
    return row;
  }
}

function percent(v: number): string {
  return `${Math.round(v * 100)}%`;
}
