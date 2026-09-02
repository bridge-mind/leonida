import { describe, expect, it } from 'vitest';
import { defaultBus, isLoopId, RECIPES } from '../../src/systems/audio/recipes';
import { isSoundId, SOUND_IDS } from '../../src/systems/audio/sounds';

describe('recipe registry', () => {
  it('has a recipe with a build function for every SoundId', () => {
    for (const id of SOUND_IDS) {
      const recipe = RECIPES[id];
      expect(recipe, `missing recipe for ${id}`).toBeDefined();
      expect(typeof recipe.build, `recipe ${id} has no build()`).toBe('function');
    }
  });

  it('has no recipes for unknown ids', () => {
    for (const key of Object.keys(RECIPES)) expect(isSoundId(key), `stray recipe key ${key}`).toBe(true);
    expect(Object.keys(RECIPES)).toHaveLength(SOUND_IDS.length);
  });

  it('keeps recipe gains and reverb sends in range', () => {
    for (const id of SOUND_IDS) {
      const r = RECIPES[id];
      if (r.gain !== undefined) expect(r.gain).toBeGreaterThan(0);
      if (r.reverb !== undefined) {
        expect(r.reverb).toBeGreaterThanOrEqual(0);
        expect(r.reverb).toBeLessThanOrEqual(1);
      }
    }
  });

  it('routes families to the expected buses', () => {
    expect(defaultBus('ui.blip')).toBe('ui');
    expect(defaultBus('ambient.beach.loop')).toBe('ambient');
    expect(defaultBus('casino.ambient.loop')).toBe('ambient');
    expect(defaultBus('vehicle.engine.loop')).toBe('engine');
    expect(defaultBus('vehicle.engine.helicopter.loop')).toBe('engine');
    expect(defaultBus('vehicle.horn')).toBe('sfx');
    expect(defaultBus('radio.synthwave.loop')).toBe('music');
    expect(defaultBus('weapon.pistol.fire')).toBe('sfx');
  });

  it('detects loop ids by suffix', () => {
    expect(isLoopId('vehicle.skid.loop')).toBe(true);
    expect(isLoopId('vehicle.horn')).toBe(false);
    const loops = SOUND_IDS.filter(isLoopId);
    expect(loops.length).toBeGreaterThan(20);
  });
});
