import RAPIER, { type Collider, type RigidBody } from '@dimforge/rapier3d-compat';
import { BoxGeometry, Group, Mesh, MeshStandardMaterial, Object3D, SpotLight, Vector3 } from 'three';
import { allocEntityId, type DamageInfo, type Entity, type IDamageable } from '../../core/entities';
import type { LoopHandle } from '../audio/api';
import { Groups } from '../../physics/Physics';
import { PoliceConfig } from './config';
import { aimAtPlayer, type PoliceContext } from './context';

const H = PoliceConfig.helicopter;
const bodyMat = new MeshStandardMaterial({ color: 0x1b2a44, roughness: 0.55, metalness: 0.35 });
const trimMat = new MeshStandardMaterial({ color: 0xe8e8ee, roughness: 0.6 });
const rotorMat = new MeshStandardMaterial({ color: 0x222222, roughness: 0.9, transparent: true, opacity: 0.55 });
const bodyGeo = new BoxGeometry(2.2, 1.6, 4.2);
const tailGeo = new BoxGeometry(0.5, 0.5, 4.5);
const finGeo = new BoxGeometry(0.15, 1.2, 0.9);
const rotorGeo = new BoxGeometry(11, 0.06, 0.5);
const tailRotorGeo = new BoxGeometry(0.06, 1.6, 0.25);
const skidGeo = new BoxGeometry(0.12, 0.12, 3);
const tmp = new Vector3();
const dir = new Vector3();

/**
 * Police helicopter: kinematic entity (no physical collisions, but a VEHICLE-group
 * collider so hitscan bullets can damage it) orbiting the player at altitude, a night
 * spotlight and a rifle gunner. At 0 HP it falls for a few seconds and explodes.
 */
export class Helicopter implements Entity, IDamageable {
  readonly id = allocEntityId();
  readonly kind = 'vehicle' as const;
  readonly object3d: Group;
  readonly position = new Vector3();
  isAlive = true;
  health: number = H.health;
  /** Set by the system when the level clears: climbs and flies away. */
  leaving = false;
  leaveTime = 0;
  private readonly body: RigidBody;
  private readonly collider: Collider;
  private readonly rotor: Mesh;
  private readonly tailRotor: Mesh;
  private readonly light: SpotLight;
  private readonly lightTarget = new Object3D();
  private readonly velocity = new Vector3();
  private orbitAngle: number;
  private heading = 0;
  private fireTimer = H.fireInterval;
  private losTimer = 0;
  private hasLos = false;
  private crashTime = 0;
  private exploded = false;
  private loop: LoopHandle | null;

  constructor(
    private readonly ctx: PoliceContext,
    spawn: Vector3,
  ) {
    const game = ctx.game;
    this.position.copy(spawn);
    this.orbitAngle = ctx.rng.range(0, Math.PI * 2);

    const g = new Group();
    g.name = 'helicopter';
    const hull = new Mesh(bodyGeo, bodyMat);
    hull.castShadow = true;
    const tail = new Mesh(tailGeo, bodyMat);
    tail.position.set(0, 0.3, -4.1);
    const fin = new Mesh(finGeo, trimMat);
    fin.position.set(0, 1.0, -6.0);
    this.rotor = new Mesh(rotorGeo, rotorMat);
    this.rotor.position.set(0, 1.1, 0);
    this.tailRotor = new Mesh(tailRotorGeo, rotorMat);
    this.tailRotor.position.set(0.3, 1.0, -6.2);
    const skidL = new Mesh(skidGeo, trimMat);
    skidL.position.set(-0.8, -1.0, 0);
    const skidR = new Mesh(skidGeo, trimMat);
    skidR.position.set(0.8, -1.0, 0);
    this.light = new SpotLight(0xfff4d6, 0, 140, H.spotlightAngle, 0.45, 1);
    this.light.position.set(0, -0.9, 1.2);
    this.light.target = this.lightTarget;
    g.add(hull, tail, fin, this.rotor, this.tailRotor, skidL, skidR, this.light);
    g.position.copy(spawn);
    this.object3d = g;
    game.worldRoot.add(g, this.lightTarget);

    this.body = game.physics.createKinematicBody(spawn);
    this.collider = game.physics.addCollider(RAPIER.ColliderDesc.cuboid(1.2, 0.9, 2.4), this.body, this, {
      groups: Groups.VEHICLE,
      filter: Groups.BULLET_BLOCKER,
    });
    this.loop = game.audio.loop('vehicle.engine.helicopter.loop', { position: this.position, pitch: H.loopPitch, bus: 'engine' });
  }

  update(dt: number): void {
    if (!this.isAlive) {
      this.crash(dt);
      return;
    }
    const ctx = this.ctx;
    const p = ctx.playerPosition();
    if (this.leaving) {
      this.leaveTime += dt;
      // Climb and put distance between the aircraft and the player.
      dir.set(this.position.x - p.x, 0, this.position.z - p.z);
      if (dir.lengthSq() < 1) dir.set(1, 0, 0);
      dir.normalize();
      tmp.set(this.position.x + dir.x * 60, this.position.y + 10, this.position.z + dir.z * 60);
      this.flyToward(tmp, dt);
      this.light.intensity = 0;
    } else {
      this.orbitAngle += H.orbitSpeed * dt;
      tmp.set(p.x + Math.cos(this.orbitAngle) * H.orbitRadius, p.y + H.altitude, p.z + Math.sin(this.orbitAngle) * H.orbitRadius);
      this.flyToward(tmp, dt);
      this.gunner(dt);
      const night = ctx.game.time.isNight;
      this.light.intensity = night && this.hasLos ? H.spotlightIntensity : 0;
      this.lightTarget.position.set(p.x, p.y + 1, p.z);
    }
    this.rotor.rotation.y += H.rotorSpeed * dt;
    this.tailRotor.rotation.x += H.rotorSpeed * 1.4 * dt;
    this.syncPose();
  }

  private flyToward(target: Vector3, dt: number): void {
    dir.subVectors(target, this.position);
    const dist = dir.length();
    if (dist > 1e-3) {
      dir.divideScalar(dist);
      const step = Math.min(dist, H.speed * dt);
      this.velocity.copy(dir).multiplyScalar(step / Math.max(dt, 1e-6));
      this.position.addScaledVector(dir, step);
      const flat = Math.hypot(dir.x, dir.z);
      if (flat > 0.2) {
        const desired = Math.atan2(dir.x, dir.z);
        let d = desired - this.heading;
        d = Math.atan2(Math.sin(d), Math.cos(d));
        this.heading += d * Math.min(1, dt * 2.5);
      }
    }
  }

  private gunner(dt: number): void {
    const ctx = this.ctx;
    this.losTimer -= dt;
    if (this.losTimer <= 0) {
      this.losTimer = H.losInterval;
      this.hasLos = ctx.playerAlive() && ctx.sight.toPlayer(this.position, H.losRange, this);
      if (this.hasLos) ctx.reportSighting('helicopter');
    }
    if (ctx.holdFire || !ctx.tier.lethal) return;
    this.fireTimer -= dt;
    if (this.fireTimer > 0 || !this.hasLos) return;
    this.fireTimer = H.fireInterval;
    const weapons = ctx.game.weapons;
    if (!weapons) return;
    tmp.set(this.position.x, this.position.y - 1.2, this.position.z);
    aimAtPlayer(ctx, tmp, dir);
    weapons.npcFire(this, 'rifle', tmp, dir, H.accuracy);
  }

  private crash(dt: number): void {
    this.crashTime += dt;
    this.position.y -= H.fallSpeed * dt * Math.min(2, this.crashTime + 0.5);
    this.position.addScaledVector(this.velocity, dt * 0.6);
    this.heading += dt * 2.5;
    this.object3d.rotation.z += dt * 0.8;
    this.rotor.rotation.y += H.rotorSpeed * 0.5 * dt;
    this.light.intensity = 0;
    const ground = this.ctx.game.physics.groundHeightAt(this.position.x, this.position.z, this.position.y + 2, 60);
    if ((ground !== null && this.position.y <= ground + 1) || this.crashTime > H.crashSeconds) this.explodeNow();
    this.syncPose();
  }

  private explodeNow(): void {
    if (this.exploded) return;
    this.exploded = true;
    const game = this.ctx.game;
    game.damage?.explode(this.position, H.explosionRadius, H.explosionDamage, H.explosionImpulse, this);
    game.particles?.emit('explosion', this.position, { scale: 1.6 });
    game.audio.play('explosion.large', { position: this.position });
    this.object3d.visible = false;
  }

  /** True once the wreck has blown up and can be removed. */
  get finished(): boolean {
    return this.exploded;
  }

  private syncPose(): void {
    this.object3d.position.copy(this.position);
    this.object3d.rotation.y = this.heading;
    this.body.setNextKinematicTranslation(this.position);
    if (this.loop) this.ctx.game.audio.setLoopParams(this.loop, { position: this.position });
  }

  applyDamage(info: DamageInfo): void {
    if (!this.isAlive) return;
    this.health -= info.amount;
    this.ctx.game.particles?.emit('sparks', info.point ?? this.position);
    if (info.source === 'player') {
      this.ctx.game.wanted?.reportCrime('attack_helicopter', { x: this.position.x, y: this.position.y, z: this.position.z }, 'player');
    }
    if (this.health <= 0) {
      this.health = 0;
      this.isAlive = false;
      this.velocity.multiplyScalar(0.5);
      this.ctx.game.particles?.emit('fire', this.position, { scale: 1.5 });
      if (this.loop) {
        this.ctx.game.audio.stop(this.loop);
        this.loop = null;
      }
    }
  }

  dispose(): void {
    const game = this.ctx.game;
    if (this.loop) {
      game.audio.stop(this.loop);
      this.loop = null;
    }
    game.worldRoot.remove(this.object3d, this.lightTarget);
    game.physics.removeCollider(this.collider);
    game.physics.removeBody(this.body);
    this.light.dispose();
  }
}
