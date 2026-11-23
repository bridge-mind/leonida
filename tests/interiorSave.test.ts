import { Group } from 'three';
import { describe, expect, it } from 'vitest';
import { Events } from '../src/core/Events';
import { createGameState } from '../src/core/GameState';
import type { Game } from '../src/core/Game';
import { InteriorsSystem } from '../src/systems/interiors';

describe('interior save restore', () => {
  it('reopens the saved interior without moving the player', () => {
    if (typeof document === 'undefined') {
      Object.assign(globalThis, { document: { getElementById: () => null } });
    }
    const worldRoot = new Group();
    const state = createGameState();
    const events = new Events();
    const entered: string[] = [];
    events.on('interior:entered', (e) => entered.push(e.id));
    const position = { x: 960, y: 600, z: 2.1 };
    const game = {
      worldRoot,
      interiorRoot: new Group(),
      state,
      events,
      physics: {},
      systems: { has: () => false },
      player: {
        position,
        teleport: () => {
          throw new Error('restore must not teleport');
        },
      },
    };
    const interiors = new InteriorsSystem(game as unknown as Game);
    interiors.register({
      id: 'safehouse',
      spawnPoint: { x: 1, y: 0, z: 2 },
      build: () => ({ root: new Group(), dispose() {} }),
    });

    expect(interiors.deserialize({ id: 'safehouse' })).toBeUndefined();
    expect(interiors.current).toBe('safehouse');
    expect(state.interior).toBe('safehouse');
    expect(worldRoot.visible).toBe(false);
    expect(entered).toEqual(['safehouse']);
    expect(position).toEqual({ x: 960, y: 600, z: 2.1 });
    expect(interiors.serialize()).toEqual({ id: 'safehouse' });
  });
});
