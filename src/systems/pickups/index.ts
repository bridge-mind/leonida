import type { Entity, V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { IPickupsPeer, PickupKind, WeaponId } from '../../core/peers';
import { Pool } from '../../core/Pool';
import { Random } from '../../core/Random';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { Pickup } from '../../entities/Pickup';
import type { SoundId } from '../audio/sounds';
import { PickupConfig } from './config';
import { meshKindOf, PickupRenderer } from './PickupRenderer';

const C = PickupConfig;
const WEAPON_IDS: readonly WeaponId[] = ['fists', 'pistol', 'smg', 'shotgun', 'rifle', 'sniper'];
const COLLECT_SOUND: Record<PickupKind, SoundId> = {
  cash: 'pickup.cash',
  weapon: 'pickup.weapon',
  ammo: 'pickup.ammo',
  armor: 'pickup.armor',
  health: 'pickup.health',
};

/**
 * Pooled instanced pickups with bob/spin, a magnet radius and a collect radius against
 * the player. Collecting routes to economy / player / weapons when those are loaded.
 */
export class PickupSystem implements GameSystem, IPickupsPeer {
  readonly live: Pickup[] = [];
  private renderer!: PickupRenderer;
  private readonly pool = new Pool<Pickup>(() => new Pickup());
  private readonly rng = Random.unseeded();

  constructor(private readonly game: Game) {}

  init(): void {
    this.renderer = new PickupRenderer(this.game.actorsRoot);
  }

  get count(): number {
    return this.live.length;
  }

  // --- IPickupsPeer -----------------------------------------------------------------

  spawn(kind: PickupKind, position: V3, value: number, weaponId?: WeaponId): Entity {
    if (this.live.length >= C.maxLive) this.despawn(this.oldest());
    const meshKind = meshKindOf(kind);
    let slot = this.renderer.acquire(meshKind);
    if (slot < 0) {
      const victim = this.oldestOfMesh(meshKind);
      if (victim) this.despawn(victim);
      slot = this.renderer.acquire(meshKind);
    }
    const ground = this.game.physics.groundHeightAt(position.x, position.z, position.y + 2, 8);
    const y = (ground ?? position.y) + C.hoverHeight;
    const pickup = this.pool.acquire();
    pickup.spawn(kind, { x: position.x, y, z: position.z }, Math.max(0, Math.round(value)), weaponId, this.rng.range(0, Math.PI * 2));
    pickup.instanceIndex = slot;
    this.live.push(pickup);
    return pickup;
  }

  despawn(pickup: Pickup | null): void {
    if (!pickup || !pickup.active) return;
    this.renderer.release(meshKindOf(pickup.pickupKind), pickup.instanceIndex);
    const idx = this.live.indexOf(pickup);
    if (idx >= 0) {
      this.live[idx] = this.live[this.live.length - 1];
      this.live.pop();
    }
    pickup.despawn();
    this.pool.release(pickup);
  }

  /** Live pickups within `radius` of `pos`, written into `out` (cleared first). */
  query(pos: V3, radius: number, out: Pickup[]): Pickup[] {
    out.length = 0;
    const r2 = radius * radius;
    for (let i = 0; i < this.live.length; i++) {
      const p = this.live[i];
      const dx = p.position.x - pos.x;
      const dz = p.position.z - pos.z;
      if (dx * dx + dz * dz <= r2) out.push(p);
    }
    return out;
  }

  private oldest(): Pickup | null {
    let best: Pickup | null = null;
    for (let i = 0; i < this.live.length; i++) if (!best || this.live[i].age > best.age) best = this.live[i];
    return best;
  }

  private oldestOfMesh(meshKind: ReturnType<typeof meshKindOf>): Pickup | null {
    let best: Pickup | null = null;
    for (let i = 0; i < this.live.length; i++) {
      const p = this.live[i];
      if (meshKindOf(p.pickupKind) === meshKind && (!best || p.age > best.age)) best = p;
    }
    return best;
  }

  // --- Loop -------------------------------------------------------------------------

  update(dt: number): void {
    const hasPlayer = this.game.systems.has('player');
    const player = hasPlayer ? this.game.player : undefined;
    const canCollect = player !== undefined && player.isAlive && !player.isMounted;
    const px = player ? player.position.x : 0;
    const py = player ? player.position.y : 0;
    const pz = player ? player.position.z : 0;
    for (let i = this.live.length - 1; i >= 0; i--) {
      const p = this.live[i];
      p.age += dt;
      if (p.age >= C.lifetime) {
        this.despawn(p);
        continue;
      }
      if (canCollect) {
        const dx = px - p.position.x;
        const dz = pz - p.position.z;
        const d = Math.sqrt(dx * dx + dz * dz);
        if (d <= C.collectRadius) {
          this.collect(p);
          continue;
        }
        if (d <= C.magnetRadius) {
          const step = Math.min(d, C.magnetSpeed * dt);
          p.position.x += (dx / d) * step;
          p.position.z += (dz / d) * step;
          p.baseY += (py + C.hoverHeight - p.baseY) * Math.min(1, dt * 6);
        }
      }
      p.position.y = p.baseY + Math.sin(p.age * C.bobFrequency + p.spin) * C.bobAmplitude;
      const remaining = C.lifetime - p.age;
      const blink = remaining < C.blinkBefore && Math.sin(remaining * 12) < 0;
      this.renderer.write(meshKindOf(p.pickupKind), p.instanceIndex, p.position.x, p.position.y, p.position.z, p.spin + p.age * C.spinSpeed, blink ? 0.6 : 1);
    }
    this.renderer.commit();
    this.game.perf.set('pickups', this.live.length);
  }

  private collect(p: Pickup): void {
    const game = this.game;
    switch (p.pickupKind) {
      case 'cash':
        game.economy?.add(p.value, 'pickup');
        break;
      case 'health':
        game.player.heal(p.value);
        break;
      case 'armor':
        game.player.addArmor(p.value);
        break;
      case 'weapon':
        if (p.weaponId) game.weapons?.give(p.weaponId, p.value);
        break;
      case 'ammo':
        if (p.weaponId) game.weapons?.addAmmo(p.weaponId, p.value);
        break;
    }
    game.audio.play(COLLECT_SOUND[p.pickupKind], { position: p.position });
    if (p.pickupKind === 'cash') game.particles?.emit('cashBurst', p.position);
    game.events.emit('pickup:collected', { kind: p.pickupKind, value: p.value, weaponId: p.weaponId });
    this.despawn(p);
  }

  dispose(): void {
    for (let i = this.live.length - 1; i >= 0; i--) this.despawn(this.live[i]);
    this.renderer.dispose();
  }

  // --- Debug ------------------------------------------------------------------------

  debugActions = {
    /** `pickups.spawn:<kind>:<value>[:<weaponId>]` two meters in front of the player. */
    spawn: (kind: unknown, value: unknown, weaponId?: unknown): number => {
      const player = this.game.player;
      const h = player.heading;
      const pos = { x: player.position.x + Math.sin(h) * 2, y: player.position.y, z: player.position.z + Math.cos(h) * 2 };
      const k = String(kind) as PickupKind;
      const kindOk = k === 'cash' || k === 'weapon' || k === 'ammo' || k === 'armor' || k === 'health';
      const w = WEAPON_IDS.find((id) => id === String(weaponId));
      return this.spawn(kindOk ? k : 'cash', pos, Number(value ?? 50), w ?? (k === 'weapon' || k === 'ammo' ? 'pistol' : undefined)).id;
    },
    count: (): number => this.live.length,
    clear: (): void => {
      for (let i = this.live.length - 1; i >= 0; i--) this.despawn(this.live[i]);
    },
  };
}

const module: SystemModule = {
  id: 'pickups',
  priority: PRIORITY.pickups,
  create: (game) => new PickupSystem(game),
};

export default module;
