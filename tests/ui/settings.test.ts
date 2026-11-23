import { afterEach, describe, expect, it } from 'vitest';
import { Config } from '../../src/core/Config';
import type { Game } from '../../src/core/Game';
import { applySettings, DEFAULT_SETTINGS } from '../../src/systems/ui/settings';
import { WeaponsConfig } from '../../src/systems/weapons/config';

const sensitivity = Config.camera.mouseSensitivity;
const aimAssist = WeaponsConfig.aimAssist;

function game(): Game {
  return {
    audio: { setBusVolume() {} },
    renderer: { shadowMap: { enabled: DEFAULT_SETTINGS.shadows } },
    scene: { traverse() {} },
  } as unknown as Game;
}

afterEach(() => {
  Config.camera.mouseSensitivity = sensitivity;
  WeaponsConfig.aimAssist = aimAssist;
});

describe('applySettings', () => {
  it('turns aim assist off when the pause-menu toggle says so', () => {
    WeaponsConfig.aimAssist = true;
    applySettings(game(), { ...DEFAULT_SETTINGS, aimAssist: false });
    expect(WeaponsConfig.aimAssist).toBe(false);
  });

  it('turns aim assist back on', () => {
    WeaponsConfig.aimAssist = false;
    applySettings(game(), { ...DEFAULT_SETTINGS, aimAssist: true });
    expect(WeaponsConfig.aimAssist).toBe(true);
  });
});
