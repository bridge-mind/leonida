import type { IPed, V3 } from '../../core/entities';
import type { CrimeType } from '../../core/Events';
import type { Game } from '../../core/Game';
import type { WantedState } from '../../core/GameState';
import type { IWantedPeer } from '../../core/peers';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { CRIME_TABLE } from '../../data/police';
import { WantedConfig } from './config';
import { subscribeCrimes, witnessesOf } from './crimes';
import { WantedMachine, type WantedHooks, type WitnessInfo } from './machine';
import { CopVision } from './vision';
import './events';

const ZERO: V3 = { x: 0, y: 0, z: 0 };

interface WantedSave {
  heat: number;
}

/**
 * Wanted level: crimes → stars, GTA-6-style `responding` / `active` / `searching`
 * states driven by cop line of sight, search circle + evasion timer, hot scenes and a
 * slow-decaying heat profile. Publishes to `game.state.wanted` every frame; the police
 * system reads the level and drives units.
 */
export class WantedSystem implements GameSystem, IWantedPeer {
  readonly machine: WantedMachine;
  private readonly vision: CopVision;
  private readonly scratch: IPed[] = [];
  private readonly witness: WitnessInfo = { copSaw: false, civilianSaw: false };
  private readonly offs: Array<() => void> = [];

  constructor(private readonly game: Game) {
    this.vision = new CopVision(game);
    const hooks: WantedHooks = {
      changed: (level, state, previousLevel) => this.onChanged(level, state, previousLevel),
      cleared: (reason) => this.onCleared(reason),
      starsGained: () => this.game.audio.play('wanted.star.gain', { bus: 'ui' }),
      searching: () => this.game.audio.play('wanted.searching', { bus: 'ui' }),
      hotSceneReraised: () => this.game.audio.play('wanted.hotscene', { bus: 'ui' }),
    };
    this.machine = new WantedMachine(hooks);
  }

  // --- IWantedPeer -------------------------------------------------------------------

  get level(): number {
    return this.machine.level;
  }

  get state(): WantedState {
    return this.machine.state;
  }

  get heat(): number {
    return this.machine.heat;
  }

  get playerSeen(): boolean {
    return this.machine.level > 0 && this.vision.seen(this.machine.time);
  }

  reportCrime(type: CrimeType, position: V3, actor: 'player' | 'companion'): void {
    if (actor !== 'player' && actor !== 'companion') return;
    this.handleCrime(type, position, false);
  }

  setLevel(level: number, state?: WantedState): void {
    this.machine.setLevel(level, state);
    if (this.machine.level > 0 && this.machine.state === 'active') this.vision.markSeen(this.machine.time);
    else this.vision.reset();
  }

  clear(reason: string): void {
    if (this.machine.level === 0) return;
    this.machine.clear(reason);
  }

  // --- Lifecycle ------------------------------------------------------------------------

  init(): void {
    const ev = this.game.events;
    this.offs.push(
      subscribeCrimes(this.game, {
        crime: (type, position, loud) => this.handleCrime(type, position, loud),
        phoneReport: (position) => this.machine.reportedByPhone(position),
      }),
      ev.on('wanted:sighting', () => this.vision.markSeen(this.machine.time)),
      ev.on('player:respawned', () => this.clear('respawned')),
    );
    this.publish();
  }

  fixedUpdate(dt: number): void {
    const playerPos = this.playerPosition();
    const m = this.machine;
    if (m.level > 0) this.vision.update(dt, m.time, playerPos);
    const seen = m.level > 0 && this.vision.seen(m.time) && this.playerAlive();
    m.tick(dt, playerPos, seen);
    this.publish();
  }

  dispose(): void {
    this.offs.forEach((off) => off());
    this.offs.length = 0;
  }

  serialize(): WantedSave {
    return { heat: this.machine.heat };
  }

  deserialize(data: unknown): void {
    const d = data as Partial<WantedSave> | null;
    if (d && typeof d.heat === 'number') this.machine.heat = Math.max(0, Math.min(WantedConfig.heat.max, d.heat));
    if (this.machine.level > 0) this.machine.clear('loaded');
  }

  // --- Internals --------------------------------------------------------------------------

  private handleCrime(type: CrimeType, position: V3, loud: boolean): void {
    if (!(type in CRIME_TABLE)) return;
    const witness = witnessesOf(this.game, position, loud, this.scratch, this.witness);
    const result = this.machine.reportCrime(type, position, witness);
    // A cop who saw it is looking straight at the player: start the sight clock now.
    if (result.counted && witness.copSaw) this.vision.markSeen(this.machine.time);
  }

  private onChanged(level: number, state: WantedState, previousLevel: number): void {
    this.publish();
    this.game.events.emit('wanted:changed', { level, state, previousLevel });
  }

  private onCleared(reason: string): void {
    this.vision.reset();
    this.publish();
    if (reason === 'evaded' || reason === 'respray') this.game.audio.play('wanted.lost', { bus: 'ui' });
    this.game.events.emit('wanted:cleared', { reason });
  }

  private publish(): void {
    const m = this.machine;
    const w = this.game.state.wanted;
    w.level = m.level;
    w.state = m.state;
    w.searchCenter = m.level > 0 && m.hasSearchCenter && m.state !== 'active' ? m.searchCenter : null;
    w.searchRadius = w.searchCenter ? m.searchRadius : 0;
    // The HUD reads plain {position, radius}; reuse the machine's records to avoid per-frame copies.
    const scenes = w.hotScenes;
    scenes.length = m.hotScenes.length;
    for (let i = 0; i < m.hotScenes.length; i++) scenes[i] = m.hotScenes[i];
  }

  private playerPosition(): V3 {
    return this.game.systems.has('player') ? this.game.player.position : ZERO;
  }

  private playerAlive(): boolean {
    return this.game.systems.has('player') && this.game.player.entity.isAlive;
  }

  debugActions: Record<string, (...args: unknown[]) => unknown> = {
    set: (level: unknown, state?: unknown): number => {
      this.setLevel(Number(level), isState(state) ? state : undefined);
      return this.machine.level;
    },
    clear: (): void => this.clear('debug'),
    state: (): { level: number; state: WantedState; heat: number; seen: boolean; evasion: number; evasionSeconds: number; searchRadius: number; hotScenes: number } => ({
      level: this.machine.level,
      state: this.machine.state,
      heat: this.machine.heat,
      seen: this.playerSeen,
      evasion: this.machine.evasion,
      evasionSeconds: this.machine.evasionSeconds,
      searchRadius: this.machine.searchRadius,
      hotScenes: this.machine.hotScenes.length,
    }),
    crime: (type: unknown, loud?: unknown): number => {
      const t = String(type) as CrimeType;
      if (!(t in CRIME_TABLE)) throw new Error(`unknown crime '${String(type)}'`);
      const p = this.playerPosition();
      this.game.events.emit('crime:committed', { type: t, position: { x: p.x, y: p.y, z: p.z }, actor: 'player', loud: loud === true || loud === 'true' });
      return this.machine.level;
    },
    /** Advances the evasion timer (autotests avoid waiting level×20+20 s). */
    fastForward: (seconds: unknown): number => {
      this.vision.reset();
      this.machine.fastForward(Number(seconds));
      return this.machine.level;
    },
    /** Marks the player as seen right now (as if a cop had line of sight). */
    seen: (): void => {
      this.vision.markSeen(this.machine.time);
    },
    heat: (value: unknown): number => {
      this.machine.heat = Math.max(0, Math.min(WantedConfig.heat.max, Number(value)));
      return this.machine.heat;
    },
  };
}

function isState(v: unknown): v is WantedState {
  return v === 'none' || v === 'responding' || v === 'searching' || v === 'active';
}

const module: SystemModule = {
  id: 'wanted',
  priority: PRIORITY.wanted,
  create: (game) => new WantedSystem(game),
};

export default module;
