import { afterEach, describe, expect, it } from 'vitest';
import { Config } from '../../src/core/Config';
import { Events } from '../../src/core/Events';
import type { Game } from '../../src/core/Game';
import { AudioConfig } from '../../src/systems/audio/config';
import { AudioSystem } from '../../src/systems/audio';
import { DEFAULT_SETTINGS } from '../../src/systems/ui/settings';
import { WeaponsConfig } from '../../src/systems/weapons/config';

const sensitivity = Config.camera.mouseSensitivity;
const aimAssist = WeaponsConfig.aimAssist;

function installWindow(): void {
  const g = globalThis as unknown as { window: Window };
  g.window = { addEventListener() {}, removeEventListener() {} } as unknown as Window;
}

afterEach(() => {
  Config.camera.mouseSensitivity = sensitivity;
  WeaponsConfig.aimAssist = aimAssist;
});

describe('audio boot volumes', () => {
  it('adopts the pause-menu volumes instead of the audio defaults', () => {
    installWindow();
    const events = new Events();
    let audio!: AudioSystem;
    const game = {
      events,
      time: { realElapsed: 0 },
      perf: { set() {} },
      state: { playerPos: { x: 0, y: 0, z: 0 } },
      renderer: { shadowMap: { enabled: DEFAULT_SETTINGS.shadows } },
      scene: { traverse() {} },
      get audio() {
        return audio;
      },
    } as unknown as Game;
    audio = new AudioSystem(game);
    expect(audio.getBusVolume('master')).toBe(AudioConfig.defaultBusVolumes.master);
    expect(audio.getBusVolume('music')).toBe(AudioConfig.defaultBusVolumes.music);
    audio.init();
    expect(audio.getBusVolume('master')).toBe(DEFAULT_SETTINGS.masterVolume);
    expect(audio.getBusVolume('music')).toBe(DEFAULT_SETTINGS.musicVolume);
    expect(audio.getBusVolume('sfx')).toBe(DEFAULT_SETTINGS.sfxVolume);
    expect(audio.getBusVolume('engine')).toBe(AudioConfig.defaultBusVolumes.engine);
    audio.dispose();
  });
});
