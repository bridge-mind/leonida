import type { Game } from '../../core/Game';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { NullAudio, type AudioApi } from './api';
import { SOUND_IDS } from './sounds';

/**
 * Audio API stub (the one intentional stub in the foundation, see docs/progress/foundation.md).
 * Implements the full `AudioApi` as silent no-ops so every other system can call
 * `game.audio.*` today. Agent 7 replaces this file with the procedural Web Audio
 * implementation while keeping `AudioApi` and `sounds.ts` intact.
 */
export class AudioSystem extends NullAudio implements GameSystem, AudioApi {
  constructor(private readonly game: Game) {
    super();
  }

  init(): void {
    this.game.perf.set('sounds', SOUND_IDS.length);
  }

  debugActions = {
    /** Plays every known sound id once (validates the union). */
    playAll: (): number => {
      for (const id of SOUND_IDS) this.play(id);
      return SOUND_IDS.length;
    },
  };
}

const module: SystemModule = {
  id: 'audio',
  priority: PRIORITY.audio,
  create: (game) => new AudioSystem(game),
};

export default module;
