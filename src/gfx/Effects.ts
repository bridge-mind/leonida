import {
  AdditiveBlending,
  CanvasTexture,
  LinearFilter,
  NormalBlending,
  PointLight,
  type Scene,
  ShaderMaterial,
  Sprite,
  SpriteMaterial,
  type Texture,
} from 'three';
import type { V3 } from '../core/entities';

/**
 * Shared visual-effect building blocks: the particle sprite atlas, the point-sprite
 * shader, and the muzzle flash (sprite + one reusable point light).
 */

function makeCanvas(w: number, h: number): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas unavailable');
  return { canvas, ctx };
}

function softDisc(ctx: CanvasRenderingContext2D, x: number, y: number, size: number): void {
  const r = size / 2;
  const g = ctx.createRadialGradient(x + r, y + r, 0, x + r, y + r, r);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.35, 'rgba(255,255,255,0.85)');
  g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(x, y, size, size);
}

function toTexture(canvas: HTMLCanvasElement): Texture {
  const tex = new CanvasTexture(canvas);
  tex.minFilter = LinearFilter;
  tex.magFilter = LinearFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

/** Two 64×64 tiles side by side: a soft radial disc (shape 0) and a hard square (shape 1). */
export function particleSpriteTexture(): Texture {
  const { canvas, ctx } = makeCanvas(128, 64);
  softDisc(ctx, 0, 0, 64);
  ctx.fillStyle = 'rgba(255,255,255,1)';
  ctx.fillRect(64 + 10, 10, 44, 44);
  ctx.fillStyle = 'rgba(255,255,255,0.55)';
  ctx.fillRect(64 + 14, 14, 36, 36);
  return toTexture(canvas);
}

/** Single soft disc with a bright core, used by the muzzle flash sprite. */
export function flashTexture(): Texture {
  const { canvas, ctx } = makeCanvas(64, 64);
  softDisc(ctx, 0, 0, 64);
  softDisc(ctx, 16, 16, 32);
  return toTexture(canvas);
}

const POINT_VERTEX = /* glsl */ `
attribute float size;
attribute float alpha;
attribute float shape;
attribute vec3 color;
uniform float uHeight;
varying vec4 vColor;
varying float vShape;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  // World-size points: pixels = size * (viewportHeight/2) / (tan(fov/2) * depth).
  gl_PointSize = size * uHeight * 0.5 * projectionMatrix[1][1] / max(0.05, -mv.z);
  vColor = vec4(color, alpha);
  vShape = shape;
}`;

const POINT_FRAGMENT = /* glsl */ `
uniform sampler2D uMap;
varying vec4 vColor;
varying float vShape;
void main() {
  vec2 uv = vec2((gl_PointCoord.x + vShape) * 0.5, 1.0 - gl_PointCoord.y);
  float a = texture2D(uMap, uv).a * vColor.a;
  if (a < 0.004) discard;
  gl_FragColor = vec4(vColor.rgb, a);
  #include <colorspace_fragment>
}`;

/** Point-sprite material with per-particle size/color/alpha/shape attributes. */
export function particlePointMaterial(map: Texture, additive: boolean): ShaderMaterial {
  return new ShaderMaterial({
    uniforms: { uMap: { value: map }, uHeight: { value: 1080 } },
    vertexShader: POINT_VERTEX,
    fragmentShader: POINT_FRAGMENT,
    transparent: true,
    depthWrite: false,
    depthTest: true,
    blending: additive ? AdditiveBlending : NormalBlending,
  });
}

export interface MuzzleFlashOptions {
  /** Seconds the flash stays lit. */
  duration: number;
  lightIntensity: number;
  lightDistance: number;
  spriteSize: number;
}

/**
 * One sprite + one point light, re-used for every shot so the scene's light count never
 * changes (adding a light at runtime would recompile every material).
 */
export class MuzzleFlash {
  private readonly sprite: Sprite;
  private readonly light: PointLight;
  private timer = 0;

  constructor(
    scene: Scene,
    private readonly opts: MuzzleFlashOptions,
  ) {
    const mat = new SpriteMaterial({
      map: flashTexture(),
      color: 0xfff0b0,
      blending: AdditiveBlending,
      depthWrite: false,
      transparent: true,
      toneMapped: false,
    });
    this.sprite = new Sprite(mat);
    this.sprite.scale.setScalar(opts.spriteSize);
    this.sprite.visible = false;
    this.sprite.frustumCulled = false;
    this.light = new PointLight(0xffc060, 0, opts.lightDistance, 2);
    this.light.castShadow = false;
    scene.add(this.sprite, this.light);
  }

  flash(position: V3, direction: V3): void {
    // Push the sprite slightly forward so it is not clipped by the gun hand / car window.
    this.sprite.position.set(position.x + direction.x * 0.15, position.y + direction.y * 0.15, position.z + direction.z * 0.15);
    this.sprite.material.rotation = Math.random() * Math.PI * 2;
    this.sprite.scale.setScalar(this.opts.spriteSize * (0.8 + Math.random() * 0.5));
    this.sprite.visible = true;
    this.light.position.set(position.x, position.y + 0.05, position.z);
    this.light.intensity = this.opts.lightIntensity;
    this.timer = this.opts.duration;
  }

  update(dt: number): void {
    if (this.timer <= 0) return;
    this.timer -= dt;
    if (this.timer <= 0) {
      this.sprite.visible = false;
      this.light.intensity = 0;
    } else {
      const k = this.timer / this.opts.duration;
      this.light.intensity = this.opts.lightIntensity * k;
    }
  }

  get active(): boolean {
    return this.timer > 0;
  }

  dispose(scene: Scene): void {
    scene.remove(this.sprite, this.light);
    this.sprite.material.map?.dispose();
    this.sprite.material.dispose();
    this.light.dispose();
  }
}
