import type { V3 } from '../../core/entities';
import type { CrimeType } from '../../core/Events';
import type { WantedState } from '../../core/GameState';
import { CRIME_TABLE, MAX_WANTED_LEVEL } from '../../data/police';
import { WantedConfig } from './config';

export interface HotScene {
  position: V3;
  radius: number;
  /** Machine time at which the scene expires. */
  expiresAt: number;
  /** Re-raising needs an outside→inside transition, so the flag starts true when created under the player. */
  playerInside: boolean;
}

export interface WitnessInfo {
  copSaw: boolean;
  civilianSaw: boolean;
}

export interface CrimeResult {
  counted: boolean;
  starsAdded: number;
}

/** Callbacks the machine fires; the system maps them to events, audio and state writes. */
export interface WantedHooks {
  changed(level: number, state: WantedState, previousLevel: number): void;
  cleared(reason: string): void;
  starsGained(count: number): void;
  searching(): void;
  hotSceneReraised(): void;
}

const C = WantedConfig;

/**
 * Pure wanted-level state machine (no engine dependencies, unit-testable):
 * crime table → stars, `responding` / `active` / `searching` transitions driven by
 * cop line of sight, the growing search circle, evasion timer, hot scenes and heat.
 * Time is whatever the caller feeds into `tick` (simulation seconds).
 */
export class WantedMachine {
  level = 0;
  state: WantedState = 'none';
  heat = 0;
  readonly searchCenter: V3 = { x: 0, y: 0, z: 0 };
  hasSearchCenter = false;
  searchRadius = 0;
  readonly lastSeenPos: V3 = { x: 0, y: 0, z: 0 };
  lastSeenAt = -Infinity;
  /** Seconds of successful evasion accumulated since the last sighting / crime. */
  evasion = 0;
  time = 0;
  readonly hotScenes: HotScene[] = [];

  private searchTime = 0;
  private readonly playerPos: V3 = { x: 0, y: 0, z: 0 };
  private readonly cooldownUntil: Partial<Record<CrimeType, number>> = {};
  private readonly scenePool: HotScene[] = [];

  constructor(private readonly hooks: WantedHooks) {}

  /** Seconds the player must stay unseen (outside the circle) for the level to clear. */
  get evasionSeconds(): number {
    return C.evasion.baseSeconds + C.evasion.perLevelSeconds * this.level;
  }

  get maxSearchRadius(): number {
    return C.search.maxRadiusBase + C.search.maxRadiusPerLevel * this.level;
  }

  // --- Per-frame -------------------------------------------------------------------

  tick(dt: number, playerPos: V3, seen: boolean): void {
    this.time += dt;
    this.playerPos.x = playerPos.x;
    this.playerPos.y = playerPos.y;
    this.playerPos.z = playerPos.z;
    if (this.heat > 0) this.heat = Math.max(0, this.heat - (C.heat.decayPerMinute / 60) * dt);
    this.updateHotScenes();
    if (this.level === 0) return;

    const prevLevel = this.level;
    const prevState = this.state;
    if (seen) {
      this.lastSeenAt = this.time;
      this.lastSeenPos.x = playerPos.x;
      this.lastSeenPos.y = playerPos.y;
      this.lastSeenPos.z = playerPos.z;
      this.evasion = 0;
      this.state = 'active';
    } else {
      if (this.state === 'active' && this.time - this.lastSeenAt > C.vision.loseSightAfter) {
        this.state = 'searching';
        this.setSearchCenter(this.lastSeenPos);
        this.evasion = 0;
        this.hooks.searching();
      }
      if (this.state === 'searching' || this.state === 'responding') {
        this.searchTime += dt;
        this.searchRadius = Math.min(C.search.initialRadius + C.search.growthPerSecond * this.searchTime, this.maxSearchRadius);
        const inside = distance2(playerPos, this.searchCenter) < this.searchRadius * this.searchRadius;
        this.evasion += inside ? dt / C.evasion.hiddenInsideMultiplier : dt;
        if (this.evasion >= this.evasionSeconds) {
          this.clear('evaded');
          return;
        }
      }
    }
    this.commit(prevLevel, prevState);
  }

  /** Debug / autotests: advance the evasion timer without waiting. */
  fastForward(seconds: number): void {
    if (this.level === 0) return;
    if (this.state === 'active') {
      // Force the sight loss first so the fast-forward lands in `searching`.
      this.lastSeenAt = -Infinity;
    }
    const prevLevel = this.level;
    const prevState = this.state;
    if (this.state === 'active') {
      this.state = 'searching';
      this.setSearchCenter(this.lastSeenPos);
      this.hooks.searching();
    }
    this.searchTime += seconds;
    this.searchRadius = Math.min(C.search.initialRadius + C.search.growthPerSecond * this.searchTime, this.maxSearchRadius);
    this.evasion += seconds;
    if (this.evasion >= this.evasionSeconds) {
      this.clear('evaded');
      return;
    }
    this.commit(prevLevel, prevState);
  }

  // --- Crimes ------------------------------------------------------------------------

  reportCrime(type: CrimeType, position: V3, witness: WitnessInfo): CrimeResult {
    const def = CRIME_TABLE[type];
    const until = this.cooldownUntil[type];
    if (until !== undefined && this.time < until) return { counted: false, starsAdded: 0 };
    const witnessed = def.witness === 'none' || witness.copSaw || witness.civilianSaw || this.level > 0;
    if (!witnessed) return { counted: false, starsAdded: 0 };
    if (def.cooldown > 0) this.cooldownUntil[type] = this.time + def.cooldown;

    const prevLevel = this.level;
    const prevState = this.state;
    const stars = prevLevel === 0 ? def.stars : def.starsWhenWanted;
    this.level = Math.min(MAX_WANTED_LEVEL, prevLevel + stars);
    if (this.level === 0) return { counted: false, starsAdded: 0 };

    if (witness.copSaw) {
      this.lastSeenAt = this.time;
      this.lastSeenPos.x = this.playerPos.x;
      this.lastSeenPos.y = this.playerPos.y;
      this.lastSeenPos.z = this.playerPos.z;
      this.state = 'active';
    } else if (this.state !== 'active') {
      this.state = 'responding';
      this.setSearchCenter(position);
    }
    this.evasion = 0;
    if (def.hotScene) this.addHotScene(position);
    const gained = this.level - prevLevel;
    if (gained > 0) {
      this.heat = Math.min(C.heat.max, this.heat + C.heat.perStar * gained);
      this.hooks.starsGained(gained);
    }
    this.commit(prevLevel, prevState);
    return { counted: true, starsAdded: gained };
  }

  /** A civilian finished calling it in: police head to the scene without knowing who did it. */
  reportedByPhone(position: V3): void {
    const prevLevel = this.level;
    const prevState = this.state;
    if (this.level === 0) {
      this.level = 1;
      this.heat = Math.min(C.heat.max, this.heat + C.heat.perStar);
      this.hooks.starsGained(1);
    }
    if (this.state !== 'active') {
      this.state = 'responding';
      this.setSearchCenter(position);
    }
    this.evasion = 0;
    this.addHotScene(position);
    this.commit(prevLevel, prevState);
  }

  // --- Direct control ------------------------------------------------------------------

  setLevel(level: number, state?: WantedState): void {
    const target = Math.max(0, Math.min(MAX_WANTED_LEVEL, Math.round(level)));
    if (target === 0) {
      this.clear('debug');
      return;
    }
    const prevLevel = this.level;
    const prevState = this.state;
    this.level = target;
    this.state = state && state !== 'none' ? state : 'active';
    if (this.state === 'active') {
      this.lastSeenAt = this.time;
      this.lastSeenPos.x = this.playerPos.x;
      this.lastSeenPos.y = this.playerPos.y;
      this.lastSeenPos.z = this.playerPos.z;
    } else {
      this.setSearchCenter(this.playerPos);
    }
    this.evasion = 0;
    if (target > prevLevel) {
      this.heat = Math.min(C.heat.max, this.heat + C.heat.perStar * (target - prevLevel));
      this.hooks.starsGained(target - prevLevel);
    }
    this.commit(prevLevel, prevState);
  }

  clear(reason: string): void {
    const prevLevel = this.level;
    this.level = 0;
    this.state = 'none';
    this.hasSearchCenter = false;
    this.searchRadius = 0;
    this.searchTime = 0;
    this.evasion = 0;
    this.lastSeenAt = -Infinity;
    // Scenes the player is standing in must be left and re-entered before they re-raise.
    for (let i = 0; i < this.hotScenes.length; i++) {
      const s = this.hotScenes[i];
      s.playerInside = distance2(this.playerPos, s.position) < s.radius * s.radius;
    }
    this.hooks.cleared(reason);
    if (prevLevel !== 0) this.hooks.changed(0, 'none', prevLevel);
  }

  // --- Hot scenes -----------------------------------------------------------------------

  private addHotScene(position: V3): void {
    const H = C.hotScene;
    for (let i = 0; i < this.hotScenes.length; i++) {
      const s = this.hotScenes[i];
      if (distance2(position, s.position) < H.mergeRadius * H.mergeRadius) {
        s.expiresAt = this.time + H.expirySeconds;
        return;
      }
    }
    if (this.hotScenes.length >= H.max) this.recycleScene(0);
    const scene = this.scenePool.pop() ?? { position: { x: 0, y: 0, z: 0 }, radius: H.radius, expiresAt: 0, playerInside: true };
    scene.position.x = position.x;
    scene.position.y = position.y;
    scene.position.z = position.z;
    scene.radius = H.radius;
    scene.expiresAt = this.time + H.expirySeconds;
    scene.playerInside = distance2(this.playerPos, position) < H.radius * H.radius;
    this.hotScenes.push(scene);
  }

  private recycleScene(index: number): void {
    const s = this.hotScenes[index];
    this.hotScenes[index] = this.hotScenes[this.hotScenes.length - 1];
    this.hotScenes.pop();
    this.scenePool.push(s);
  }

  private updateHotScenes(): void {
    for (let i = this.hotScenes.length - 1; i >= 0; i--) {
      const s = this.hotScenes[i];
      if (this.time >= s.expiresAt) {
        this.recycleScene(i);
        continue;
      }
      const inside = distance2(this.playerPos, s.position) < s.radius * s.radius;
      if (inside && !s.playerInside && this.level === 0) {
        s.playerInside = true;
        const prevLevel = this.level;
        const prevState = this.state;
        this.level = C.hotScene.reraiseLevel;
        this.state = 'searching';
        this.setSearchCenter(s.position);
        this.evasion = 0;
        this.hooks.hotSceneReraised();
        this.hooks.starsGained(this.level);
        this.commit(prevLevel, prevState);
        continue;
      }
      s.playerInside = inside;
    }
  }

  // --- Helpers ----------------------------------------------------------------------------

  private setSearchCenter(p: V3): void {
    this.searchCenter.x = p.x;
    this.searchCenter.y = p.y;
    this.searchCenter.z = p.z;
    this.hasSearchCenter = true;
    this.searchRadius = C.search.initialRadius;
    this.searchTime = 0;
  }

  private commit(prevLevel: number, prevState: WantedState): void {
    if (this.level !== prevLevel || this.state !== prevState) this.hooks.changed(this.level, this.state, prevLevel);
  }
}

function distance2(a: V3, b: V3): number {
  const dx = a.x - b.x;
  const dz = a.z - b.z;
  return dx * dx + dz * dz;
}
