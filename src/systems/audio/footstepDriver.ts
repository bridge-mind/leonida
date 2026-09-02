import { Vector3 } from 'three';
import type { Game } from '../../core/Game';
import { Groups, type RaycastHit } from '../../physics/Physics';
import type { PlayerSystem } from '../player';
import { isStaticEntity, type StaticSurface } from '../../world/city/StaticEntities';
import type { AudioApi } from './api';
import { AudioConfig } from './config';
import type { SoundId } from './sounds';

const F = AudioConfig.footsteps;
const DOWN = new Vector3(0, -1, 0);

type FootSurface = 'concrete' | 'asphalt' | 'grass' | 'sand' | 'dirt' | 'wood' | 'metal' | 'water';

const SURFACE_MAP: Record<StaticSurface, FootSurface> = {
  concrete: 'concrete',
  asphalt: 'asphalt',
  sand: 'sand',
  grass: 'grass',
  dirt: 'dirt',
  wood: 'wood',
  metal: 'metal',
  water: 'water',
  glass: 'concrete',
};

/**
 * Plays footsteps for the player from a stride accumulator: steps per second =
 * speed / stride, stride 0.75 m walking → 1.4 m sprinting. The ground surface is
 * classified by a cheap downward raycast every 0.5 s. Also plays jump/land cues
 * from grounded transitions.
 */
export class FootstepDriver {
  private stride = 0;
  private rayTimer = F.surfaceRayInterval;
  private surface: FootSurface = 'concrete';
  private wasGrounded = true;
  private fallSpeed = 0;
  private readonly origin = new Vector3();
  private readonly hit: RaycastHit = {
    entity: null,
    point: new Vector3(),
    normal: new Vector3(),
    toi: 0,
    collider: null as unknown as RaycastHit['collider'],
  };

  constructor(
    private readonly game: Game,
    private readonly audio: AudioApi,
  ) {}

  update(dt: number): void {
    const player = this.game.systems.get<PlayerSystem>('player');
    if (!player || !player.isAlive || player.isMounted) {
      this.stride = 0;
      this.wasGrounded = true;
      return;
    }
    const v = player.velocity;
    this.trackAirtime(player, v.y);
    const speed = Math.hypot(v.x, v.z);
    if (speed < F.minSpeed || !player.entity.grounded) {
      this.stride = 0;
      return;
    }
    this.rayTimer += dt;
    if (this.rayTimer >= F.surfaceRayInterval) {
      this.rayTimer = 0;
      this.classify(player);
    }
    const speed01 = clamp01((speed - F.walkSpeed) / (F.sprintSpeed - F.walkSpeed));
    const strideLength = F.strideWalk + (F.strideSprint - F.strideWalk) * speed01;
    this.stride += speed * dt;
    if (this.stride >= strideLength) {
      this.stride -= strideLength;
      const volume = F.volumeMin + (F.volumeMax - F.volumeMin) * speed01;
      const pitch = 1 + (Math.random() * 2 - 1) * F.pitchJitter;
      this.audio.play(`footstep.${this.surface}` as SoundId, { position: player.position, volume, pitch });
    }
  }

  /** Jump on leaving the ground upward, land (hard above the fall-speed threshold) on touching down. */
  private trackAirtime(player: PlayerSystem, vy: number): void {
    const grounded = player.entity.grounded;
    if (this.wasGrounded && !grounded) {
      this.fallSpeed = 0;
      if (vy > 1) this.audio.play('player.jump', { position: player.position, volume: 0.6 });
    } else if (!grounded) {
      this.fallSpeed = Math.max(this.fallSpeed, -vy);
    } else if (!this.wasGrounded && this.fallSpeed > F.landMinFallSpeed) {
      const hard = this.fallSpeed > F.hardLandFallSpeed;
      this.audio.play(hard ? 'player.land.hard' : 'player.land', { position: player.position, volume: hard ? 1 : 0.7 });
      this.stride = 0;
    }
    this.wasGrounded = grounded;
  }

  private classify(player: PlayerSystem): void {
    const world = this.game.systems.has('world') ? this.game.world : undefined;
    if (world && world.isInWater(player.position)) {
      this.surface = 'water';
      return;
    }
    const p = player.position;
    this.origin.set(p.x, p.y + F.rayHeight, p.z);
    const hit = this.game.physics.raycastInto(this.hit, this.origin, DOWN, F.rayLength, { groups: Groups.STATIC, exclude: player.entity });
    if (hit && isStaticEntity(hit.entity)) this.surface = SURFACE_MAP[hit.entity.surface];
  }
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
