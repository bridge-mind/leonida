import { Color, DirectionalLight, Fog, HemisphereLight, MathUtils, Vector3 } from 'three';
import { Sky } from 'three/addons/objects/Sky.js';
import { Config } from '../core/Config';
import type { Game } from '../core/Game';

const DN = Config.dayNight;

interface ColorKey {
  e: number;
  c: [number, number, number];
}

const HORIZON_KEYS: ColorKey[] = [
  { e: -0.35, c: [0.015, 0.02, 0.05] },
  { e: -0.12, c: [0.08, 0.05, 0.14] },
  { e: -0.03, c: [0.42, 0.2, 0.38] },
  { e: 0.04, c: [0.98, 0.5, 0.28] },
  { e: 0.14, c: [0.98, 0.72, 0.5] },
  { e: 0.35, c: [0.86, 0.84, 0.82] },
  { e: 0.7, c: [0.68, 0.8, 0.94] },
];

const SUN_KEYS: ColorKey[] = [
  { e: -0.05, c: [1.0, 0.35, 0.18] },
  { e: 0.05, c: [1.0, 0.55, 0.28] },
  { e: 0.18, c: [1.0, 0.8, 0.55] },
  { e: 0.45, c: [1.0, 0.97, 0.9] },
];

function sampleKeys(keys: ColorKey[], e: number, out: Color): Color {
  if (e <= keys[0].e) return out.setRGB(...keys[0].c);
  for (let i = 0; i + 1 < keys.length; i++) {
    if (e <= keys[i + 1].e) {
      const t = MathUtils.smoothstep(e, keys[i].e, keys[i + 1].e);
      const a = keys[i].c;
      const b = keys[i + 1].c;
      return out.setRGB(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t);
    }
  }
  const last = keys[keys.length - 1].c;
  return out.setRGB(...last);
}

/**
 * Sun/moon, sky dome, fog and ambient light driven by the game clock. Exposes
 * `night01` (0 = lamps off, 1 = lamps on) that the world uses for windows, lamps and neon.
 */
export class DayNight {
  readonly sun: DirectionalLight;
  readonly hemi: HemisphereLight;
  readonly sky: Sky;
  readonly fog: Fog;
  readonly sunDirection = new Vector3(0, 1, 0);
  /** Sine of the sun elevation (negative below the horizon). */
  sunElevation = 0;
  night01 = 0;
  readonly horizonColor = new Color();
  private readonly sunColor = new Color();
  private readonly tmpColor = new Color();

  constructor(private readonly game: Game) {
    this.sun = new DirectionalLight(0xffffff, DN.sunIntensityDay);
    this.sun.castShadow = Config.render.shadows;
    const sh = this.sun.shadow;
    sh.mapSize.set(Config.render.shadowMapSize, Config.render.shadowMapSize);
    sh.camera.near = 1;
    sh.camera.far = 500;
    sh.camera.left = -Config.render.shadowHalfExtent;
    sh.camera.right = Config.render.shadowHalfExtent;
    sh.camera.top = Config.render.shadowHalfExtent;
    sh.camera.bottom = -Config.render.shadowHalfExtent;
    sh.camera.updateProjectionMatrix();
    sh.bias = -0.0006;
    sh.normalBias = 0.03;
    game.scene.add(this.sun, this.sun.target);

    this.hemi = new HemisphereLight(0xbfd8ff, 0x4a4030, DN.hemiIntensityDay);
    game.scene.add(this.hemi);

    this.sky = new Sky();
    this.sky.scale.setScalar(20000);
    this.sky.name = 'sky';
    game.scene.add(this.sky);

    this.fog = new Fog(0xffffff, DN.fogNear, DN.fogFar);
    game.scene.fog = this.fog;
    this.update(0, null);
  }

  /** Hours as a 0..1 fraction of the daylight arc; <0 or >1 at night. */
  private dayFraction(hour: number): number {
    return (hour - DN.sunriseHour) / (DN.sunsetHour - DN.sunriseHour);
  }

  update(_dt: number, focus: Vector3 | null): void {
    const hour = this.game.time.hour;
    const f = this.dayFraction(hour);
    const isDay = f >= 0 && f <= 1;
    let elevAngle: number;
    let theta: number;
    if (isDay) {
      theta = Math.PI * f;
      elevAngle = Math.sin(theta) * 1.15;
    } else {
      // Night arc for the moon: wraps sunset → sunrise.
      const nightLen = 24 - (DN.sunsetHour - DN.sunriseHour);
      const since = hour > DN.sunsetHour ? hour - DN.sunsetHour : hour + 24 - DN.sunsetHour;
      theta = Math.PI * (since / nightLen);
      elevAngle = Math.sin(theta) * 0.9;
    }
    // Sun-below-horizon elevation for lighting blends (approximate: how far past dusk we are).
    const sunElevAngle = isDay ? elevAngle : -Math.min(0.6, Math.sin(theta) * 0.6);
    this.sunElevation = Math.sin(sunElevAngle);
    const e = this.sunElevation;

    const cosE = Math.cos(elevAngle);
    this.sunDirection.set(Math.cos(theta) * cosE, Math.sin(elevAngle), 0.45 * Math.sin(theta) * cosE).normalize();

    this.night01 = 1 - MathUtils.smoothstep(e, DN.lampsOnElevation, DN.lampsOffElevation);

    // Directional light: sun by day, moon by night (same light, retargeted).
    if (isDay) {
      sampleKeys(SUN_KEYS, e, this.sunColor);
      this.sun.color.copy(this.sunColor);
      // A setting sun is still bright; only fade it in the last few degrees.
      this.sun.intensity = DN.sunIntensityDay * MathUtils.clamp(0.45 + e / 0.2, 0.2, 1);
    } else {
      this.sun.color.setRGB(0.55, 0.65, 1.0);
      this.sun.intensity = DN.moonIntensity * MathUtils.clamp(Math.sin(theta) * 1.5, 0.3, 1);
    }
    const target = focus ?? this.sun.target.position;
    this.sun.target.position.copy(target);
    this.sun.position.copy(target).addScaledVector(this.sunDirection, 160);

    // Sky dome and fog.
    const u = this.sky.material.uniforms;
    (u.sunPosition.value as Vector3).copy(this.sunDirection);
    const duskness = 1 - MathUtils.smoothstep(e, 0.0, 0.3);
    u.turbidity.value = isDay ? 3 + 5 * duskness : 2;
    u.rayleigh.value = isDay ? 1.2 + 2.3 * duskness : 0.35;
    u.mieCoefficient.value = isDay ? 0.004 + 0.02 * duskness : 0.002;
    u.mieDirectionalG.value = isDay ? 0.8 + 0.13 * duskness : 0.7;

    sampleKeys(HORIZON_KEYS, e, this.horizonColor);
    this.fog.color.copy(this.horizonColor);
    this.fog.near = DN.fogNear;
    this.fog.far = DN.fogFar * (isDay ? 1 : 0.75);

    // Hemisphere: sky tint above, warm ground bounce below; deep blue at night.
    const dayAmount = MathUtils.smoothstep(e, -0.12, 0.2);
    this.hemi.intensity = MathUtils.lerp(DN.hemiIntensityNight, DN.hemiIntensityDay, dayAmount);
    this.tmpColor.setRGB(0.3, 0.38, 0.65).lerp(this.horizonColor, 0.3 + 0.35 * dayAmount);
    this.hemi.color.copy(this.tmpColor);
    this.hemi.groundColor.setRGB(0.12, 0.1, 0.14).lerp(this.tmpColor.setRGB(0.55, 0.45, 0.35), dayAmount);
  }

  dispose(): void {
    this.game.scene.remove(this.sun, this.sun.target, this.hemi, this.sky);
    this.sky.material.dispose();
    this.sky.geometry.dispose();
    this.sun.dispose();
    this.hemi.dispose();
    this.game.scene.fog = null;
  }
}
