/**
 * Cross-system contracts. Each system class `implements` its peer interface and
 * other systems consume it through `game.systems.get<IXxxPeer>('xxx')`, which is
 * undefined when the peer is not loaded (isolation runs) or has been disabled.
 *
 * Only what another system needs belongs here; keep implementation details in the
 * owning system. Ids are shared string unions so data tables and events agree.
 */
import type { Vector3 } from 'three';
import type { DamageInfo, Entity, IPed, IVehicle, PedBrain, V3, VehicleDriver } from './entities';
import type { CrimeType } from './Events';
import type { Blip, WantedState } from './GameState';

export type VehicleArchetypeId = 'sedan' | 'sports' | 'muscle' | 'suv' | 'pickup' | 'van' | 'taxi' | 'police' | 'swat';

export type WeaponId = 'fists' | 'pistol' | 'smg' | 'shotgun' | 'rifle' | 'sniper';

export type PickupKind = 'cash' | 'weapon' | 'ammo' | 'armor' | 'health';

export type ParticleEffectId =
  | 'muzzleFlash'
  | 'tracer'
  | 'impactConcrete'
  | 'impactMetal'
  | 'impactDirt'
  | 'impactWater'
  | 'impactGlass'
  | 'blood'
  | 'sparks'
  | 'dust'
  | 'smoke'
  | 'fire'
  | 'explosion'
  | 'cashBurst'
  | 'tireSmoke';

export type CardKind = 'wasted' | 'busted' | 'missionPassed' | 'missionFailed' | 'custom';

export type CharacterId = 'lucia' | 'jason';

export interface DialogueLine {
  speaker: string;
  text: string;
  /** Seconds to hold the line (default: reading time by length). */
  seconds?: number;
}

// --- Stage 2 --------------------------------------------------------------------

export interface IVehiclesPeer {
  spawn(
    archetypeId: VehicleArchetypeId,
    position: V3,
    heading: number,
    opts?: { color?: number; driver?: VehicleDriver | null; parked?: boolean },
  ): IVehicle;
  despawn(vehicle: IVehicle): void;
  /** Nearest vehicle the player could enter within `radius` (empty or carjackable). */
  nearestEnterable(pos: V3, radius: number): IVehicle | null;
  /** Fills `out` (cleared first) with vehicles within `radius`; no allocation. */
  query(pos: V3, radius: number, out: IVehicle[]): IVehicle[];
  readonly all: readonly IVehicle[];
  /** Vehicle the player currently occupies, if any. */
  readonly playerVehicle: IVehicle | null;
  /** Player enters the nearest enterable vehicle (same as pressing F). Returns the vehicle. */
  enterNearest(): IVehicle | null;
  /** Player leaves the current vehicle (same as pressing F while driving). */
  exitCurrent(): void;
}

export interface ITrafficPeer {
  setDensity(count: number): void;
  /** Traffic within `radius` stops, drivers bail out and flee (gunfire, explosions). */
  panicNear(pos: V3, radius: number): void;
  /** Whether travelling through `nodeId` along `edgeId` currently has a green light. */
  isGreen(nodeId: number, edgeId: number): boolean;
  readonly count: number;
}

export interface IPedsPeer {
  spawn(typeId: string, position: V3, opts?: { brain?: PedBrain; heading?: number; weaponId?: WeaponId }): IPed;
  despawn(ped: IPed): void;
  query(pos: V3, radius: number, out: IPed[], filter?: (p: IPed) => boolean): IPed[];
  /** Alive peds that could see a crime at `pos` (alive, not cowering in a vehicle, within radius). */
  witnessesNear(pos: V3, radius: number, out: IPed[]): IPed[];
  readonly all: readonly IPed[];
  /** Ped type ids available in `data/peds.ts` (e.g. 'beachgoer', 'cop', 'clerk'). */
  readonly typeIds: readonly string[];
}

export interface IPickupsPeer {
  spawn(kind: PickupKind, position: V3, value: number, weaponId?: WeaponId): Entity;
}

export interface IWeaponsPeer {
  give(weaponId: WeaponId, ammo: number): void;
  equip(weaponId: WeaponId): void;
  addAmmo(weaponId: WeaponId, amount: number): void;
  has(weaponId: WeaponId): boolean;
  readonly current: WeaponId;
  readonly isAiming: boolean;
  /** Entity the player is currently aiming at (crosshair raycast), if any. */
  readonly aimTarget: Entity | null;
  /** Fires a hitscan shot for an NPC. `accuracy` 0..1 scales spread; 1 = perfect. */
  npcFire(shooter: Entity, weaponId: WeaponId, origin: V3, direction: V3, accuracy: number): void;
  /** Total inventory for save/stats: weapon id -> reserve ammo. */
  inventory(): Partial<Record<WeaponId, number>>;
}

export interface IDamagePeer {
  /** Single entry point for all damage; resolves headshots, armor, kill attribution and crimes. */
  apply(target: Entity, info: DamageInfo): void;
  /** Radius damage with linear falloff plus impulses on dynamic bodies. */
  explode(position: V3, radius: number, damage: number, impulse: number, source?: Entity | 'player'): void;
}

export interface IParticlesPeer {
  emit(id: ParticleEffectId, position: V3, opts?: { direction?: V3; end?: V3; scale?: number; color?: number }): void;
}

export interface INavigationPeer {
  setWaypoint(pos: V3 | null): void;
  readonly waypoint: V3 | null;
  addBlip(blip: Blip): void;
  removeBlip(id: string): void;
  /** Road route (world points) from the player to `pos`; empty when unreachable. */
  routeTo(pos: V3): Vector3[];
}

export interface IUiPeer {
  notify(text: string, kind?: 'info' | 'money' | 'warning'): void;
  /** Letterbox + slow-mo card; resolves when it has faded out. */
  card(kind: CardKind, text: string, seconds?: number): Promise<void>;
  dialogue: {
    show(lines: DialogueLine[]): Promise<void>;
    hide(): void;
    readonly active: boolean;
  };
  prompt(key: string, text: string): void;
  clearPrompt(key: string): void;
  setObjective(text: string | null): void;
  progressBar: { show(label: string, fraction: number): void; hide(): void };
  letterbox(on: boolean): void;
  pause(): void;
  resume(): void;
  readonly isPaused: boolean;
  /** True while the title screen, pause menu or another modal owns input. */
  readonly isModal: boolean;
}

export interface ICasinoPeer {
  readonly chips: number;
  buyChips(amount: number): boolean;
  cashOut(amount: number): boolean;
  /** Opens a table UI (player must be inside the casino interior). */
  play(game: 'blackjack' | 'roulette' | 'slots' | 'wheel'): void;
}

// --- Stage 3 --------------------------------------------------------------------

export interface IWantedPeer {
  readonly level: number;
  readonly state: WantedState;
  /** 0..100 criminal profile. */
  readonly heat: number;
  reportCrime(type: CrimeType, position: V3, actor: 'player' | 'companion'): void;
  /** Debug / missions: force a level (state defaults to 'active'). */
  setLevel(level: number, state?: WantedState): void;
  /** Clears stars and search state (respray, death, arrest, mission end). */
  clear(reason: string): void;
  /** True when any cop currently has line of sight to the player. */
  readonly playerSeen: boolean;
}

export interface IPolicePeer {
  readonly activeUnits: number;
  /** Fills `out` with cop peds (foot and in cruisers) within `radius`. */
  copsNear(pos: V3, radius: number, out: IPed[]): IPed[];
}

export interface IShopsPeer {
  /** Opens the Bullet Barn UI (player must be inside the interior). */
  openGunStore(): void;
  /** Store ids that are closed for the day after a robbery. */
  isStoreClosed(storeId: string): boolean;
}

export interface IRobberyPeer {
  /** Store id of the robbery in progress, if any. */
  readonly activeStore: string | null;
}

export interface IChopShopPeer {
  /** Today's 2x bonus archetype. */
  readonly wantedArchetype: VehicleArchetypeId;
  valueOf(vehicle: IVehicle): number;
}

// --- Stage 4 --------------------------------------------------------------------

export interface MissionSummary {
  id: string;
  name: string;
  contact: string;
  /** Contact marker position (world) where the mission starts. */
  position: V3;
}

export interface IMissionsPeer {
  readonly active: string | null;
  readonly completed: readonly string[];
  available(): MissionSummary[];
  start(id: string): boolean;
  abort(): void;
}

export interface ICharactersPeer {
  readonly active: CharacterId;
  switchTo(id: CharacterId): Promise<void>;
  readonly abilityCharge: number;
  activateAbility(): boolean;
}
