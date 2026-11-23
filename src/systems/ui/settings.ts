import { Config } from '../../core/Config';
import type { Game } from '../../core/Game';
import { WeaponsConfig } from '../weapons/config';
import { UiConfig } from './config';

/**
 * Player settings persisted at `localStorage['leonida.settings']`. Other systems read
 * `aimAssist` / `shadows` from here (or from storage) instead of owning their own copy.
 */
export interface Settings {
  mouseSensitivity: number;
  invertY: boolean;
  masterVolume: number;
  sfxVolume: number;
  musicVolume: number;
  aimAssist: boolean;
  shadows: boolean;
  drawDistance: number;
  bloom: boolean;
}

export const DEFAULT_SETTINGS: Settings = {
  mouseSensitivity: Config.camera.mouseSensitivity,
  invertY: false,
  masterVolume: 1,
  sfxVolume: 1,
  musicVolume: 0.8,
  aimAssist: true,
  shadows: Config.render.shadows,
  drawDistance: 1,
  bloom: false,
};

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(UiConfig.settings.storageKey);
    if (!raw) return { ...DEFAULT_SETTINGS };
    const parsed = JSON.parse(raw) as Partial<Settings>;
    return {
      mouseSensitivity: num(parsed.mouseSensitivity, DEFAULT_SETTINGS.mouseSensitivity),
      invertY: typeof parsed.invertY === 'boolean' ? parsed.invertY : DEFAULT_SETTINGS.invertY,
      masterVolume: num(parsed.masterVolume, DEFAULT_SETTINGS.masterVolume),
      sfxVolume: num(parsed.sfxVolume, DEFAULT_SETTINGS.sfxVolume),
      musicVolume: num(parsed.musicVolume, DEFAULT_SETTINGS.musicVolume),
      aimAssist: typeof parsed.aimAssist === 'boolean' ? parsed.aimAssist : DEFAULT_SETTINGS.aimAssist,
      shadows: typeof parsed.shadows === 'boolean' ? parsed.shadows : DEFAULT_SETTINGS.shadows,
      drawDistance: num(parsed.drawDistance, DEFAULT_SETTINGS.drawDistance),
      bloom: typeof parsed.bloom === 'boolean' ? parsed.bloom : DEFAULT_SETTINGS.bloom,
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
  Config.camera.invertY = s.invertY;
  WeaponsConfig.aimAssist = s.aimAssist;
  const { drawMin, drawMax } = UiConfig.settings;
  Config.render.drawDistance = Math.min(drawMax, Math.max(drawMin, s.drawDistance));
  Config.render.bloom = s.bloom;
  const far = Config.render.far * Config.render.drawDistance;
  if (game.viewCamera.far !== far) {
    game.viewCamera.far = far;
    game.viewCamera.updateProjectionMatrix();
  }
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
