import { Config } from '../../core/Config';
import type { Game } from '../../core/Game';
import { UiConfig } from './config';

/**
 * Player settings persisted at `localStorage['leonida.settings']`. Other systems read
 * `aimAssist` / `shadows` from here (or from storage) instead of owning their own copy.
 */
export interface Settings {
  mouseSensitivity: number;
  masterVolume: number;
  sfxVolume: number;
  musicVolume: number;
  aimAssist: boolean;
  shadows: boolean;
}

export const DEFAULT_SETTINGS: Settings = {
  mouseSensitivity: Config.camera.mouseSensitivity,
  masterVolume: 1,
  sfxVolume: 1,
  musicVolume: 0.8,
  aimAssist: true,
  shadows: Config.render.shadows,
};

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(UiConfig.settings.storageKey);
    if (!raw) return { ...DEFAULT_SETTINGS };
    const parsed = JSON.parse(raw) as Partial<Settings>;
    return {
      mouseSensitivity: num(parsed.mouseSensitivity, DEFAULT_SETTINGS.mouseSensitivity),
      masterVolume: num(parsed.masterVolume, DEFAULT_SETTINGS.masterVolume),
      sfxVolume: num(parsed.sfxVolume, DEFAULT_SETTINGS.sfxVolume),
      musicVolume: num(parsed.musicVolume, DEFAULT_SETTINGS.musicVolume),
      aimAssist: typeof parsed.aimAssist === 'boolean' ? parsed.aimAssist : DEFAULT_SETTINGS.aimAssist,
      shadows: typeof parsed.shadows === 'boolean' ? parsed.shadows : DEFAULT_SETTINGS.shadows,
    };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(settings: Settings): void {
  try {
    localStorage.setItem(UiConfig.settings.storageKey, JSON.stringify(settings));
  } catch {
    /* storage unavailable (private mode); settings stay in memory */
  }
}

/** Pushes the values into the live systems that consume them. */
export function applySettings(game: Game, s: Settings): void {
  const { sensitivityMin, sensitivityMax } = UiConfig.settings;
  Config.camera.mouseSensitivity = Math.min(sensitivityMax, Math.max(sensitivityMin, s.mouseSensitivity));
  game.audio.setBusVolume('master', s.masterVolume);
  game.audio.setBusVolume('sfx', s.sfxVolume);
  game.audio.setBusVolume('music', s.musicVolume);
  if (game.renderer.shadowMap.enabled !== s.shadows) {
    game.renderer.shadowMap.enabled = s.shadows;
    // Three caches per-material programs; forcing a recompile makes the toggle take effect.
    game.scene.traverse((o) => {
      const mesh = o as { material?: { needsUpdate?: boolean } | { needsUpdate?: boolean }[] };
      const mat = mesh.material;
      if (Array.isArray(mat)) for (const m of mat) m.needsUpdate = true;
      else if (mat) mat.needsUpdate = true;
    });
  }
}

function num(v: unknown, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}
