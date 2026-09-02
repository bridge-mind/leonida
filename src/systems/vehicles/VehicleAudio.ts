import type { V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { Vehicle } from '../../entities/Vehicle';
import type { LoopHandle } from '../audio/api';
import type { SoundId } from '../audio/sounds';
import { VehicleConfig } from './config';

type LoopKind = 'engine' | 'skid' | 'fire' | 'siren';

const LOOP_IDS: Record<LoopKind, SoundId> = {
  engine: 'vehicle.engine.loop',
  skid: 'vehicle.skid.loop',
  fire: 'vehicle.fire.loop',
  siren: 'vehicle.siren.loop',
};

const RANGE2: Record<LoopKind, number> = {
  engine: VehicleConfig.audioDistance ** 2,
  skid: VehicleConfig.audioDistance ** 2,
  fire: 120 ** 2,
  siren: 400 ** 2,
};

/** Starts/stops the looping sounds each vehicle needs based on distance and state. */
export class VehicleAudio {
  private readonly loops: Record<LoopKind, Map<Vehicle, LoopHandle>> = {
    engine: new Map(),
    skid: new Map(),
    fire: new Map(),
    siren: new Map(),
  };

  constructor(private readonly game: Game) {}

  update(vehicles: readonly Vehicle[], listener: V3): void {
    const audio = this.game.audio;
    for (let i = 0; i < vehicles.length; i++) {
      const v = vehicles[i];
      const dx = v.position.x - listener.x;
      const dz = v.position.z - listener.z;
      const d2 = dx * dx + dz * dz;
      const s = v.state;
      const running = v.isAlive && !s.wrecked && v.hasController;
      const engine = this.sync('engine', v, running && d2 < RANGE2.engine);
      if (engine) {
        audio.engine(engine, s.rpm01, v.controls.throttle, s.speedMs);
        audio.setLoopParams(engine, { position: v.position });
      }
      const sliding = running && s.grounded && Math.abs(s.lateralSpeed) > VehicleConfig.handling.skidLateralSpeed && s.speedMs > 3;
      const skid = this.sync('skid', v, sliding && d2 < RANGE2.skid);
      if (skid) audio.setLoopParams(skid, { position: v.position, volume: Math.min(1, Math.abs(s.lateralSpeed) / 12) });
      const fire = this.sync('fire', v, v.isAlive && s.onFire && d2 < RANGE2.fire);
      if (fire) audio.setLoopParams(fire, { position: v.position });
      const siren = this.sync('siren', v, v.isAlive && s.sirenOn && d2 < RANGE2.siren);
      if (siren) audio.setLoopParams(siren, { position: v.position });
    }
  }

  private sync(kind: LoopKind, v: Vehicle, wanted: boolean): LoopHandle | null {
    const map = this.loops[kind];
    let handle = map.get(v) ?? null;
    if (wanted && !handle) {
      handle = this.game.audio.loop(LOOP_IDS[kind], { position: v.position });
      if (kind === 'siren') this.game.audio.siren(handle, true);
      map.set(v, handle);
    } else if (!wanted && handle) {
      if (kind === 'siren') this.game.audio.siren(handle, false);
      this.game.audio.stop(handle);
      map.delete(v);
      handle = null;
    }
    return handle;
  }

  /** Stop everything attached to a vehicle (despawn). */
  remove(v: Vehicle): void {
    for (const kind of Object.keys(this.loops) as LoopKind[]) this.sync(kind, v, false);
  }

  dispose(): void {
    for (const map of Object.values(this.loops)) {
      for (const handle of map.values()) this.game.audio.stop(handle);
      map.clear();
    }
  }
}
