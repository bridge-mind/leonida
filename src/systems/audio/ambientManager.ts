import type { Game } from '../../core/Game';
import type { LoopHandle } from './api';
import { AudioConfig } from './config';
import { isSoundId, type SoundId } from './sounds';

export interface AmbientHost {
  loop(id: SoundId, opts: { volume: number; bus: 'ambient' }): LoopHandle;
  fadeLoop(handle: LoopHandle, volume: number, seconds: number): void;
  stop(handle: LoopHandle): void;
}

interface Bed {
  id: SoundId;
  handle: LoopHandle;
}

const A = AudioConfig.ambient;

/**
 * Picks the ambient bed from the player's district (or the interior), crossfades
 * between beds, and scales the night-crickets layer by `world.night01`.
 */
export class AmbientManager {
  private timer = 0;
  private current: Bed | null = null;
  private fading: { bed: Bed; until: number } | null = null;
  private crickets: LoopHandle | null = null;
  private elapsed = 0;

  constructor(
    private readonly game: Game,
    private readonly host: AmbientHost,
  ) {}

  get currentId(): SoundId | null {
    return this.current?.id ?? null;
  }

  update(dt: number): void {
    this.elapsed += dt;
    if (this.fading && this.elapsed >= this.fading.until) {
      this.host.stop(this.fading.bed.handle);
      this.fading = null;
    }
    this.timer += dt;
    if (this.timer < A.pollInterval) return;
    this.timer = 0;

    const game = this.game;
    const world = game.systems.has('world') ? game.world : undefined;
    const interior = game.state.interior;
    const district = world?.districtAt(game.state.playerPos) ?? null;
    let id: SoundId;
    if (interior) {
      id = interior.includes('casino') ? (A.casino as SoundId) : (A.interior as SoundId);
    } else if (district && isSoundId(district.ambient)) {
      id = district.ambient;
    } else {
      id = A.cityFallback as SoundId;
    }
    if (!this.current || this.current.id !== id) this.switchTo(id);

    const night = world ? world.night01 : 0;
    const quiet = interior !== null || (district !== null && A.noCricketDistricts.includes(district.id));
    const target = quiet ? 0 : night * A.cricketsMax;
    if (target > 0.005 || this.crickets) {
      if (!this.crickets) this.crickets = this.host.loop(A.fallback as SoundId, { volume: 0, bus: 'ambient' });
      this.host.fadeLoop(this.crickets, target, A.crossfade);
    }
  }

  private switchTo(id: SoundId): void {
    if (this.fading) {
      this.host.stop(this.fading.bed.handle);
      this.fading = null;
    }
    if (this.current) {
      this.host.fadeLoop(this.current.handle, 0, A.crossfade);
      this.fading = { bed: this.current, until: this.elapsed + A.crossfade + 0.2 };
    }
    const handle = this.host.loop(id, { volume: 0, bus: 'ambient' });
    this.host.fadeLoop(handle, A.bedVolume, A.crossfade);
    this.current = { id, handle };
  }

  dispose(): void {
    if (this.current) this.host.stop(this.current.handle);
    if (this.fading) this.host.stop(this.fading.bed.handle);
    if (this.crickets) this.host.stop(this.crickets);
    this.current = null;
    this.fading = null;
    this.crickets = null;
  }
}
