import { Vector3 } from 'three';
import { isDamageable, type BodyPart, type DamageInfo, type Entity, type V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { ParticleEffectId } from '../../core/peers';
import type { WeaponDef } from '../../data/weapons';
import { Groups, type RaycastHit } from '../../physics/Physics';
import { isStaticEntity, type StaticSurface } from '../../world/city/StaticEntities';
import type { SoundId } from '../audio/sounds';
import { isDummy } from './Dummy';
import { randomInCone } from './spread';

const DEFAULT_HEAD_HEIGHT = 1.5;

interface SurfaceFx {
  effect: ParticleEffectId;
  sound: SoundId;
}

const SURFACE_FX: Record<StaticSurface, SurfaceFx> = {
  concrete: { effect: 'impactConcrete', sound: 'impact.concrete' },
  asphalt: { effect: 'impactConcrete', sound: 'impact.asphalt' },
  sand: { effect: 'impactDirt', sound: 'impact.sand' },
  grass: { effect: 'impactDirt', sound: 'impact.dirt' },
  dirt: { effect: 'impactDirt', sound: 'impact.dirt' },
  wood: { effect: 'impactConcrete', sound: 'impact.wood' },
  metal: { effect: 'impactMetal', sound: 'impact.metal' },
  water: { effect: 'impactWater', sound: 'impact.water' },
  glass: { effect: 'impactGlass', sound: 'glass.shatter' },
};
const FLESH_FX: SurfaceFx = { effect: 'blood', sound: 'impact.flesh' };
const VEHICLE_FX: SurfaceFx = { effect: 'sparks', sound: 'impact.vehicle' };
const DUMMY_FX: SurfaceFx = { effect: 'impactConcrete', sound: 'impact.wood' };

export interface FireParams {
  def: WeaponDef;
  /** Where tracers and the flash start. */
  muzzle: V3;
  /** Where the ray starts (camera for the player, muzzle for NPCs). */
  origin: V3;
  /** Normalised aim direction. */
  direction: V3;
  /** Cone half-angle in radians. */
  halfAngle: number;
  /** Shooter's own entity (or vehicle) to ignore. */
  exclude: Entity | null;
  /** Scales `def.damage` (character abilities); undefined = 1. */
  damageMultiplier?: number;
  shooter: Entity | 'player';
  /** First projectile flies exactly along `direction` (scripted aim). */
  exactFirst: boolean;
}

/**
 * Hitscan shots shared by the player and NPCs: cone spread per pellet, raycast against
 * static/vehicle/ped groups, damage through `game.damage`, impact FX by surface, tracer,
 * muzzle flash, fire sound and `weapon:fired`.
 */
export class Hitscan {
  private readonly hit: RaycastHit = {
    entity: null,
    point: new Vector3(),
    normal: new Vector3(),
    toi: 0,
    collider: null as unknown as RaycastHit['collider'],
  };
  private readonly dir = new Vector3();
  private readonly end = new Vector3();
  /** Number of projectiles that hit a damageable entity during the last `fire`. */
  lastHits = 0;

  constructor(private readonly game: Game) {}

  fire(p: FireParams): void {
    const def = p.def;
    this.lastHits = 0;
    for (let i = 0; i < def.pellets; i++) {
      if (i === 0 && p.exactFirst) this.dir.set(p.direction.x, p.direction.y, p.direction.z);
      else randomInCone(p.direction, p.halfAngle, this.dir);
      this.castProjectile(def, p.origin, this.dir, p.exclude, p.shooter, this.end, p.damageMultiplier ?? 1);
      this.game.particles?.emit('tracer', p.muzzle, { end: this.end });
    }
    this.game.particles?.emit('muzzleFlash', p.muzzle, { direction: p.direction });
    this.game.audio.play(def.sounds.fire, { position: p.muzzle });
    // Listeners (peds hearing shots, wanted) keep these, so they get their own copies. The
    // reported origin is the muzzle: for the player the ray itself starts at the camera.
    this.game.events.emit('weapon:fired', {
      weaponId: def.id,
      origin: { x: p.muzzle.x, y: p.muzzle.y, z: p.muzzle.z },
      direction: { x: p.direction.x, y: p.direction.y, z: p.direction.z },
      shooter: p.shooter,
    });
  }

  /** Casts one projectile; writes the end point (hit or max range) to `outEnd` and returns the entity hit. */
  castProjectile(
    def: WeaponDef,
    origin: V3,
    dir: Vector3,
    exclude: Entity | null,
    shooter: Entity | 'player',
    outEnd: Vector3,
    damageMultiplier = 1,
  ): Entity | null {
    const h = this.game.physics.raycastInto(this.hit, origin, dir, def.range, {
      groups: Groups.STATIC | Groups.VEHICLE | Groups.PED,
      exclude: exclude ?? undefined,
    });
    if (!h) {
      outEnd.set(origin.x, origin.y, origin.z).addScaledVector(dir, def.range);
      return null;
    }
    outEnd.copy(h.point);
    const entity = h.entity;
    this.impactFx(entity, h.point, h.normal);
    if (entity && (entity.kind === 'player' || isDamageable(entity))) {
      const headHeight = (entity as { headHeight?: number }).headHeight ?? DEFAULT_HEAD_HEIGHT;
      const part: BodyPart = h.point.y - entity.position.y > headHeight ? 'head' : 'body';
      // Fresh objects: damage consumers may hold on to point/direction (ragdoll, decals).
      const info: DamageInfo = {
        amount: def.damage * damageMultiplier,
        type: 'bullet',
        point: { x: h.point.x, y: h.point.y, z: h.point.z },
        direction: { x: dir.x, y: dir.y, z: dir.z },
        part,
        source: shooter,
      };
      this.applyDamage(entity, info);
      this.lastHits++;
    }
    return entity;
  }

  /** Routes through the damage system when loaded so isolation runs behave the same. */
  applyDamage(entity: Entity, info: DamageInfo): void {
    const damage = this.game.damage;
    if (damage) {
      damage.apply(entity, info);
    } else if (entity.kind === 'player') {
      if (this.game.systems.has('player')) this.game.player.applyDamage(info);
    } else if (isDamageable(entity)) {
      entity.applyDamage(info);
    }
  }

  impactFx(entity: Entity | null, point: Vector3, normal: Vector3): void {
    let fx: SurfaceFx = SURFACE_FX.concrete;
    if (isStaticEntity(entity)) fx = SURFACE_FX[entity.surface];
    else if (entity && (entity.kind === 'ped' || entity.kind === 'player')) fx = FLESH_FX;
    else if (entity && entity.kind === 'vehicle') fx = VEHICLE_FX;
    else if (isDummy(entity)) fx = DUMMY_FX;
    // Peds emit their own blood puff in `Ped.applyDamage` (covers NPC fire and melee too).
    if (entity?.kind !== 'ped') this.game.particles?.emit(fx.effect, point, { direction: normal });
    this.game.audio.play(fx.sound, { position: point });
  }
}
