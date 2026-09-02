import type { V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { CharacterId, ICharactersPeer, WeaponId } from '../../core/peers';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { LANDMARK_BY_ID } from '../../data/landmarks';
import { PED_PALETTES, setPedPalette } from '../../gfx/PedBuilder';
import { AbilityFx } from '../../ui/missions/AbilityFx';
import { CharacterWheel } from '../../ui/missions/CharacterWheel';
import type { MissionSystem } from '../missions';
import { CHARACTER_IDS, CharactersConfig } from './config';

const K = CharactersConfig;

interface CharacterState {
  position: V3;
  heading: number;
  health: number;
  armor: number;
  weapons: Partial<Record<WeaponId, number>>;
  current: WeaponId;
}

interface CharactersSave {
  active: CharacterId;
  charge: number;
  states: Partial<Record<CharacterId, CharacterState>>;
}

/**
 * Lucia/Jason switching (Q wheel + sky transition) and the adrenaline ability (Caps Lock).
 * Only one Player entity exists: switching stores the leaving character's state and
 * re-dresses the entity for the other one.
 */
export class CharacterSystem implements GameSystem, ICharactersPeer {
  private current: CharacterId = 'lucia';
  private readonly states: Partial<Record<CharacterId, CharacterState>> = {};
  private charge = 1;
  private abilityTimer = 0;
  private abilityOwner: CharacterId | null = null;
  private luciaBoost = false;
  private switching = false;
  private wheel: CharacterWheel | null = null;
  private fx: AbilityFx | null = null;
  private readonly offs: (() => void)[] = [];
  private uiMounted = false;

  constructor(private readonly game: Game) {}

  init(): void {
    this.applyStats(this.current);
    this.game.state.activeCharacter = this.current;
    this.game.state.abilityCharge = this.charge;
    this.offs.push(
      this.game.events.on('weapon:fired', (e) => {
        if (this.luciaBoost && e.shooter === 'player') this.endLuciaBoost();
      }),
    );
  }

  /** UI lives under `#ui`, which the UI system (higher priority) creates after us. */
  private mountUi(): void {
    if (this.uiMounted) return;
    const parent = document.getElementById('ui');
    if (!parent) return;
    this.uiMounted = true;
    this.wheel = new CharacterWheel(
      this.game,
      parent,
      () => this.current,
      () => this.canOpenWheel(),
      (id) => void this.switchTo(id),
    );
    this.fx = new AbilityFx(this.game, parent);
  }

  // --- ICharactersPeer ---------------------------------------------------------------

  get active(): CharacterId {
    return this.current;
  }

  get abilityCharge(): number {
    return this.charge;
  }

  get isSwitching(): boolean {
    return this.switching;
  }

  get abilityActive(): boolean {
    return this.abilityTimer > 0;
  }

  async switchTo(id: CharacterId): Promise<void> {
    const game = this.game;
    if (id === this.current || this.switching || !game.systems.has('player')) return;
    const blocked = this.switchBlockReason();
    if (blocked) {
      game.ui?.notify(blocked, 'warning');
      return;
    }
    this.switching = true;
    game.input.pushBlocker(K.switch.blocker);
    try {
      const player = game.player;
      if (player.isMounted) game.vehicles?.exitCurrent();
      this.states[this.current] = this.snapshot();
      const from = player.position;
      const fromV3 = { x: from.x, y: from.y, z: from.z };
      game.audio.play('character.switch.whoosh');
      const cam = game.systems.has('camera') ? game.camera : null;
      if (cam) await cam.cinematic.moveTo({ x: from.x, y: from.y + K.switch.skyHeight, z: from.z }, fromV3, K.switch.upSeconds);

      const next = this.states[id] ?? this.defaultState(id);
      player.teleport(next.position, next.heading);
      this.dress(id);
      player.entity.health = next.health;
      player.entity.armor = next.armor;
      this.restoreWeapons(next);
      this.current = id;
      game.state.activeCharacter = id;
      game.audio.play('character.switch');

      if (cam) {
        const to = player.position;
        const h = player.heading;
        // Hard cut in the sky: the down-glide starts from the camera's current position.
        cam.position.set(to.x, to.y + K.switch.skyHeight, to.z);
        cam.lookAt.set(to.x, to.y, to.z);
        await cam.cinematic.moveTo(
          { x: to.x - Math.sin(h) * 4, y: to.y + 2, z: to.z - Math.cos(h) * 4 },
          { x: to.x, y: to.y + 1.4, z: to.z },
          K.switch.downSeconds,
        );
        cam.setOrbit(h + Math.PI, 0.18);
        cam.cinematic.stop();
      }
      game.events.emit('character:switched', { to: id });
    } finally {
      game.input.popBlocker(K.switch.blocker);
      this.switching = false;
    }
  }

  activateAbility(): boolean {
    const game = this.game;
    if (this.charge < 1 || this.abilityTimer > 0 || !game.systems.has('player') || !game.player.isAlive) return false;
    this.charge = 0;
    this.abilityTimer = K.ability.seconds;
    this.abilityOwner = this.current;
    game.time.scale = K.ability.timeScale;
    game.audio.play('ability.activate');
    this.mountUi();
    this.fx?.start(this.current);
    if (this.current === 'lucia') {
      this.luciaBoost = true;
      game.player.entity.stats.damageMultiplier = K.ability.luciaDamage;
    }
    game.events.emit('ability:activated', { character: this.current });
    return true;
  }

  private endAbility(): void {
    const game = this.game;
    this.abilityTimer = 0;
    if (game.time.scale === K.ability.timeScale) game.time.scale = 1;
    this.fx?.stop();
    game.audio.play('ability.end');
    if (this.abilityOwner) game.events.emit('ability:ended', { character: this.abilityOwner });
    this.abilityOwner = null;
  }

  private endLuciaBoost(): void {
    this.luciaBoost = false;
    if (this.game.systems.has('player')) this.game.player.entity.stats.damageMultiplier = K.stats[this.current].damageMultiplier;
  }

  // --- Helpers ------------------------------------------------------------------------

  private switchBlockReason(): string | null {
    const missions = this.game.systems.get<MissionSystem>('missions');
    if (missions?.companionRequired) return "Can't switch: your partner is with you on this mission";
    if (missions?.active) return "Can't switch during a mission";
    if (this.game.ui?.isModal) return 'Close the menu first';
    if (!this.game.player.isAlive) return null;
    return null;
  }

  private canOpenWheel(): boolean {
    const game = this.game;
    if (this.switching || !game.systems.has('player') || !game.player.isAlive) return false;
    if (game.input.isBlocked || (game.ui?.isModal ?? false) || game.state.interior) return false;
    return true;
  }

  private snapshot(): CharacterState {
    const player = this.game.player;
    const p = player.position;
    return {
      position: { x: p.x, y: p.y, z: p.z },
      heading: player.heading,
      health: player.entity.health,
      armor: player.entity.armor,
      weapons: this.game.weapons?.inventory() ?? {},
      current: this.game.weapons?.current ?? 'fists',
    };
  }

  private defaultState(id: CharacterId): CharacterState {
    const lm = LANDMARK_BY_ID[K.switch.defaultLandmark];
    const h = lm ? lm.doorHeading : 0;
    const off = K.switch.defaultDoorOffset + (id === 'jason' ? 1.5 : 0);
    const position = lm ? { x: lm.door.x + Math.sin(h) * off, y: 0, z: lm.door.z + Math.cos(h) * off } : { x: 0, y: 0, z: 0 };
    return { position, heading: h + Math.PI, health: 100, armor: 0, weapons: { pistol: 60 }, current: 'pistol' };
  }

  private dress(id: CharacterId): void {
    const entity = this.game.player.entity;
    const palette = PED_PALETTES[id];
    if (palette) setPedPalette(entity.rig, palette);
    this.applyStats(id);
  }

  private applyStats(id: CharacterId): void {
    if (!this.game.systems.has('player')) return;
    const s = K.stats[id];
    const stats = this.game.player.entity.stats;
    stats.sprintMultiplier = s.sprintMultiplier;
    stats.recoilMultiplier = s.recoilMultiplier;
    stats.damageMultiplier = this.luciaBoost && id === 'lucia' ? K.ability.luciaDamage : s.damageMultiplier;
  }

  /** No "clear inventory" API exists (see progress doc); weapons are only ever added. */
  private restoreWeapons(state: CharacterState): void {
    const weapons = this.game.weapons;
    if (!weapons) return;
    for (const [id, ammo] of Object.entries(state.weapons) as [WeaponId, number][]) {
      if (!weapons.has(id)) weapons.give(id, ammo);
    }
    if (weapons.has(state.current)) weapons.equip(state.current);
  }

  // --- Loop ---------------------------------------------------------------------------

  update(): void {
    const game = this.game;
    this.mountUi();
    const realDt = game.time.realDt;
    this.wheel?.update();
    if (this.abilityTimer > 0) {
      this.abilityTimer -= realDt;
      this.fx?.update();
      if (this.abilityTimer <= 0) this.endAbility();
    } else if (this.charge < 1) {
      this.charge = Math.min(1, this.charge + realDt / K.ability.rechargeSeconds);
      if (this.charge >= 1) game.audio.play('ability.recharged');
    }
    game.state.abilityCharge = this.charge;
    if (game.input.wasPressed(K.ability.key) && !game.input.isBlocked && !(game.ui?.isModal ?? false)) this.activateAbility();
  }

  serialize(): CharactersSave {
    if (this.game.systems.has('player')) this.states[this.current] = this.snapshot();
    return { active: this.current, charge: this.charge, states: { ...this.states } };
  }

  deserialize(data: unknown): void {
    const d = data as Partial<CharactersSave> | null;
    if (!d || (d.active !== 'lucia' && d.active !== 'jason')) return;
    for (const id of CHARACTER_IDS) {
      const s = d.states?.[id];
      if (s && typeof s.heading === 'number') this.states[id] = s;
    }
    this.charge = typeof d.charge === 'number' ? Math.max(0, Math.min(1, d.charge)) : 1;
    this.current = d.active;
    // Position/health/weapons for the active character are restored by their own systems.
    if (this.game.systems.has('player')) this.dress(this.current);
    this.game.state.activeCharacter = this.current;
    this.game.state.abilityCharge = this.charge;
  }

  dispose(): void {
    for (const off of this.offs) off();
    if (this.abilityTimer > 0) this.endAbility();
    this.wheel?.dispose();
    this.fx?.dispose();
  }

  debugActions = {
    switch: (id: unknown): Promise<void> => this.switchTo(id === 'jason' ? 'jason' : 'lucia'),
    ability: (): boolean => this.activateAbility(),
    charge: (value?: unknown): number => {
      if (value !== undefined) this.charge = Math.max(0, Math.min(1, Number(value)));
      return this.charge;
    },
    state: (): { active: CharacterId; charge: number; abilityActive: boolean; switching: boolean; wheelOpen: boolean } => ({
      active: this.current,
      charge: this.charge,
      abilityActive: this.abilityTimer > 0,
      switching: this.switching,
      wheelOpen: this.wheel?.isOpen ?? false,
    }),
  };
}

const module: SystemModule = {
  id: 'characters',
  priority: PRIORITY.characters,
  create: (game) => new CharacterSystem(game),
};

export default module;
