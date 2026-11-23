import { afterEach, describe, expect, it } from 'vitest';
import { Config } from '../../src/core/Config';
import type { Game } from '../../src/core/Game';
import { applySettings, DEFAULT_SETTINGS } from '../../src/systems/ui/settings';
import { WeaponsConfig } from '../../src/systems/weapons/config';

const snapshot = {
  sensitivity: Config.camera.mouseSensitivity,
  invertY: Config.camera.invertY,
  draw: Config.render.drawDistance,
  bloom: Config.render.bloom,
  aim: WeaponsConfig.aimAssist,
};

function fakeGame(far = Config.render.far): { game: Game; updates: { n: number } } {
  const updates = { n: 0 };
  return {
    updates,
    game: {
      audio: { setBusVolume() {} },
      renderer: { shadowMap: { enabled: DEFAULT_SETTINGS.shadows } },
      scene: { traverse() {} },
      viewCamera: {
        far,
        updateProjectionMatrix() {
          updates.n += 1;
        },
      },
    } as unknown as Game,
  };
}

afterEach(() => {
  Config.camera.mouseSensitivity = snapshot.sensitivity;
  Config.camera.invertY = snapshot.invertY;
  Config.render.drawDistance = snapshot.draw;
  Config.render.bloom = snapshot.bloom;
  WeaponsConfig.aimAssist = snapshot.aim;
});

describe('pause menu settings from the original list', () => {
  it('applies invert Y, draw distance, bloom and aim assist', () => {
    const { game, updates } = fakeGame();
    applySettings(game, {
      ...DEFAULT_SETTINGS,
      invertY: true,
      drawDistance: 0.5,
      bloom: true,
      aimAssist: false,
    });
    expect(Config.camera.invertY).toBe(true);
    expect(Config.render.drawDistance).toBe(0.5);
    expect(Config.render.bloom).toBe(true);
    expect(WeaponsConfig.aimAssist).toBe(false);
    expect(game.viewCamera.far).toBeCloseTo(Config.render.far * 0.5);
    expect(updates.n).toBe(1);
  });

  it('keeps the authored draw distance when the menu is at 100%', () => {
    const { game, updates } = fakeGame(Config.render.far);
    applySettings(game, { ...DEFAULT_SETTINGS, drawDistance: 1, bloom: false, invertY: false });
    expect(Config.render.drawDistance).toBe(1);
    expect(Config.render.bloom).toBe(false);
    expect(updates.n).toBe(0);
  });
});
