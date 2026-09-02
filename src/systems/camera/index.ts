import { MathUtils, PerspectiveCamera, Vector3 } from 'three';
import { Config } from '../../core/Config';
import type { CameraTarget, V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { Groups, type RaycastHit } from '../../physics/Physics';
import { CinematicController } from './cinematic';

export type CameraMode = 'thirdPerson' | 'aim' | 'vehicle' | 'cinematic' | 'map';

const C = Config.camera;

/**
 * GTA V-style camera. Yaw 0 looks toward -Z; forward = (-sin yaw, 0, -cos yaw).
 * Pitch is positive when the camera is above the pivot looking down.
 */
export class CameraSystem implements GameSystem {
  readonly three: PerspectiveCamera;
  mode: CameraMode = 'thirdPerson';
  yaw = Math.PI;
  pitch = 0.18;
  target: CameraTarget | null = null;
  readonly cinematic: CinematicController;

  /** Smoothed camera state. */
  readonly position = new Vector3();
  readonly lookAt = new Vector3();
  fov = C.thirdPerson.fov;
  private currentDistance = C.thirdPerson.distance;
  private shakeStrength = 0;
  private shakeTime = 0;
  private shakeDuration = 0;
  private fovKickAmount = 0;
  private mouseIdle = 0;
  private aimBlend = 0;
  private prevMode: CameraMode = 'thirdPerson';
  private initialised = false;

  private readonly pivot = new Vector3();
  private readonly desiredPos = new Vector3();
  private readonly desiredLook = new Vector3();
  private readonly dir = new Vector3();
  private readonly rayDir = new Vector3();
  private readonly shakeOffset = new Vector3();
  private readonly velDir = new Vector3();
  private readonly hit: RaycastHit = {
    entity: null,
    point: new Vector3(),
    normal: new Vector3(),
    toi: 0,
    collider: null as unknown as RaycastHit['collider'],
  };

  constructor(private readonly game: Game) {
    this.three = game.viewCamera;
    this.cinematic = new CinematicController(this);
  }

  init(): void {
    const player = this.game.systems.has('player') ? this.game.player : undefined;
    if (player) {
      this.setTarget(player.cameraTarget);
      this.yaw = player.heading + Math.PI;
    }
  }

  // --- Public API --------------------------------------------------------------

  setTarget(target: CameraTarget): void {
    this.target = target;
    if (target.kind === 'vehicle' && this.mode !== 'cinematic' && this.mode !== 'map') this.mode = 'vehicle';
    if (target.kind === 'onFoot' && this.mode === 'vehicle') this.mode = 'thirdPerson';
  }

  setMode(mode: CameraMode): void {
    if (mode === this.mode) return;
    if (mode === 'aim') {
      this.aimBlend = 0;
      this.fov = C.aim.fovStart;
    }
    if (this.mode !== 'cinematic') this.prevMode = this.mode;
    this.mode = mode;
  }

  /** Returns to the mode that was active before the last cinematic/map switch. */
  restoreMode(): void {
    this.mode = this.prevMode;
  }

  /** Set the orbit angles directly (autotests, cutscene hand-off). */
  setOrbit(yaw: number, pitch: number): void {
    this.yaw = yaw;
    this.pitch = MathUtils.clamp(pitch, C.pitchMin, C.pitchMax);
  }

  shake(strength: number, seconds: number): void {
    this.shakeStrength = Math.max(this.shakeStrength, strength);
    this.shakeDuration = Math.max(this.shakeDuration - this.shakeTime, seconds);
    this.shakeTime = 0;
  }

  fovKick(delta: number): void {
    this.fovKickAmount += delta;
  }

  /** Horizontal forward vector for the current yaw. */
  forward(out: Vector3): Vector3 {
    return out.set(-Math.sin(this.yaw), 0, -Math.cos(this.yaw));
  }

  right(out: Vector3): Vector3 {
    return out.set(Math.cos(this.yaw), 0, -Math.sin(this.yaw));
  }

  /** Snap the smoothed state to the desired state (after teleports). */
  snap(): void {
    this.initialised = false;
  }

  // --- Update ------------------------------------------------------------------

  update(_dt: number): void {
    const dt = this.game.time.realDt;
    const input = this.game.input;
    if (this.mode !== 'cinematic' && this.mode !== 'map' && !input.isBlocked) {
      const dx = input.mouseDeltaX;
      const dy = input.mouseDeltaY;
      if (dx !== 0 || dy !== 0) {
        this.yaw -= dx * C.mouseSensitivity;
        this.pitch = MathUtils.clamp(this.pitch + dy * C.mouseSensitivity, C.pitchMin, C.pitchMax);
        this.mouseIdle = 0;
      } else {
        this.mouseIdle += dt;
      }
    }

    let stiffness = C.thirdPerson.stiffness;
    let targetFov = C.thirdPerson.fov;
    switch (this.mode) {
      case 'thirdPerson':
        this.computeThirdPerson(C.thirdPerson.distance, 0, C.pivotHeight, 0.3);
        break;
      case 'aim': {
        this.aimBlend = Math.min(1, this.aimBlend + dt / 0.25);
        this.computeThirdPerson(C.aim.distance, C.aim.offsetX, C.aim.height, 0);
        stiffness = C.aim.stiffness;
        targetFov = MathUtils.lerp(C.aim.fovStart, C.aim.fovEnd, this.aimBlend);
        break;
      }
      case 'vehicle':
        targetFov = this.computeVehicle(dt);
        stiffness = C.vehicle.stiffness;
        break;
      case 'cinematic':
        this.cinematic.update(dt, this.desiredPos, this.desiredLook);
        targetFov = this.cinematic.fov;
        stiffness = 1e6;
        break;
      case 'map':
        this.computeMap();
        stiffness = 1e6;
        break;
    }

    if (!this.initialised) {
      this.position.copy(this.desiredPos);
      this.lookAt.copy(this.desiredLook);
      this.fov = targetFov;
      this.initialised = true;
    } else {
      const k = 1 - Math.exp(-stiffness * dt);
      const kLook = 1 - Math.exp(-C.thirdPerson.lookStiffness * dt);
      this.position.lerp(this.desiredPos, k);
      this.lookAt.lerp(this.desiredLook, kLook);
      this.fov += (targetFov - this.fov) * (1 - Math.exp(-8 * dt));
    }

    // Decay kicks and shake.
    this.fovKickAmount *= Math.exp(-6 * dt);
    if (this.shakeDuration > 0) {
      this.shakeTime += dt;
      const t = this.shakeTime / this.shakeDuration;
      if (t >= 1) {
        this.shakeDuration = 0;
        this.shakeStrength = 0;
        this.shakeOffset.set(0, 0, 0);
      } else {
        const env = this.shakeStrength * (1 - t) * (1 - t);
        const w = this.game.time.realElapsed * C.shakeFrequency;
        this.shakeOffset.set(Math.sin(w * 1.1) * env, Math.sin(w * 1.7 + 1.3) * env, Math.sin(w * 0.9 + 2.1) * env * 0.5);
      }
    }

    const cam = this.three;
    cam.position.copy(this.position).add(this.shakeOffset);
    cam.lookAt(this.lookAt);
    const fov = this.fov + this.fovKickAmount;
    if (Math.abs(cam.fov - fov) > 0.01) {
      cam.fov = fov;
      cam.updateProjectionMatrix();
    }
    const t = this.target;
    this.game.audio.setListener(cam.position, cam.quaternion, t ? t.velocity : ZERO);
  }

  private computeThirdPerson(distance: number, offsetX: number, pivotHeight: number, heightAbovePivot: number): void {
    const t = this.target;
    if (!t) {
      this.desiredPos.set(0, 3, 8);
      this.desiredLook.set(0, 1, 0);
      return;
    }
    this.pivot.set(t.position.x, t.position.y + pivotHeight, t.position.z);
    if (offsetX !== 0) {
      this.right(this.dir);
      this.pivot.addScaledVector(this.dir, offsetX);
    }
    const cp = Math.cos(this.pitch);
    this.dir.set(Math.sin(this.yaw) * cp, Math.sin(this.pitch), Math.cos(this.yaw) * cp);
    const allowed = this.collisionDistance(distance);
    this.currentDistance = allowed < this.currentDistance ? allowed : this.currentDistance + (allowed - this.currentDistance) * Math.min(1, this.game.time.realDt * 4);
    this.desiredPos.copy(this.pivot).addScaledVector(this.dir, this.currentDistance);
    this.desiredPos.y += heightAbovePivot * (this.currentDistance / distance);
    // Look slightly ahead of the character so it sits left of center like GTA.
    this.forward(this.rayDir);
    this.desiredLook.copy(this.pivot).addScaledVector(this.rayDir, 1.5);
  }

  private computeVehicle(dt: number): number {
    const t = this.target;
    if (!t) return C.vehicle.fovLow;
    const size = t.size ?? 4.5;
    const distance = MathUtils.clamp(C.vehicle.minDistance + (size - 4) * 0.6, C.vehicle.minDistance, C.vehicle.maxDistance);
    const speed = Math.hypot(t.velocity.x, t.velocity.z);
    // Auto-center behind the direction of travel once the mouse has been idle.
    if (this.mouseIdle > C.vehicle.autoCenterDelay && speed > 1.5) {
      this.velDir.set(t.velocity.x, 0, t.velocity.z).normalize();
      const targetYaw = Math.atan2(-this.velDir.x, -this.velDir.z);
      this.yaw = lerpAngle(this.yaw, targetYaw, 1 - Math.exp(-C.vehicle.autoCenterRate * dt));
      this.pitch += (0.15 - this.pitch) * (1 - Math.exp(-2 * dt));
    }
    this.pivot.set(t.position.x, t.position.y + 1.0, t.position.z);
    const cp = Math.cos(this.pitch);
    this.dir.set(Math.sin(this.yaw) * cp, Math.sin(this.pitch), Math.cos(this.yaw) * cp);
    const allowed = this.collisionDistance(distance);
    this.currentDistance = allowed < this.currentDistance ? allowed : this.currentDistance + (allowed - this.currentDistance) * Math.min(1, dt * 4);
    this.desiredPos.copy(this.pivot).addScaledVector(this.dir, this.currentDistance);
    this.desiredPos.y += (C.vehicle.height - 1.0) * (this.currentDistance / distance);
    this.desiredLook.copy(this.pivot);
    return MathUtils.lerp(C.vehicle.fovLow, C.vehicle.fovHigh, MathUtils.clamp(speed / C.vehicle.fovSpeedMs, 0, 1));
  }

  private computeMap(): void {
    const t = this.target;
    const x = t ? t.position.x : 0;
    const z = t ? t.position.z : 0;
    this.desiredPos.set(x, C.map.height, z + 0.01);
    this.desiredLook.set(x, 0, z);
  }

  /** Raycast from the pivot along the camera offset; returns the max safe distance. */
  private collisionDistance(distance: number): number {
    const player = this.game.systems.has('player') ? this.game.player.entity : undefined;
    // While driving, the player's own car must not push the camera in.
    const exclude = player?.mounted ? player.mounted.vehicle : player;
    const hit = this.game.physics.raycastInto(this.hit, this.pivot, this.dir, distance + C.collisionRadius, {
      groups: Groups.STATIC | Groups.VEHICLE,
      exclude,
    });
    if (!hit) return distance;
    return Math.max(0.5, hit.toi - C.collisionRadius);
  }

  debugActions = {
    setOrbit: (yaw: unknown, pitch: unknown): void => this.setOrbit(Number(yaw), Number(pitch)),
    setMode: (mode: unknown): void => this.setMode(String(mode) as CameraMode),
    shake: (strength: unknown, seconds: unknown): void => this.shake(Number(strength), Number(seconds)),
    moveTo: (x: unknown, y: unknown, z: unknown, lx: unknown, ly: unknown, lz: unknown, seconds: unknown): Promise<void> =>
      this.cinematic.moveTo(v3(x, y, z), v3(lx, ly, lz), Number(seconds ?? 0)),
    orbit: (cx: unknown, cz: unknown, radius: unknown, height: unknown, speed: unknown): void =>
      this.cinematic.orbit({ x: Number(cx), y: 0, z: Number(cz) }, Number(radius), Number(height), Number(speed)),
    restore: (): void => this.restoreMode(),
    state: (): Record<string, unknown> => ({ mode: this.mode, yaw: this.yaw, pitch: this.pitch, fov: this.fov, position: this.position.toArray() }),
  };
}

const ZERO = new Vector3();

function v3(x: unknown, y: unknown, z: unknown): V3 {
  return { x: Number(x), y: Number(y), z: Number(z) };
}

function lerpAngle(a: number, b: number, t: number): number {
  let diff = b - a;
  while (diff > Math.PI) diff -= Math.PI * 2;
  while (diff < -Math.PI) diff += Math.PI * 2;
  return a + diff * t;
}

const module: SystemModule = {
  id: 'camera',
  priority: PRIORITY.camera,
  create: (game) => new CameraSystem(game),
};

export default module;
