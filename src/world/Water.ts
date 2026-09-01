import { Color, DoubleSide, Group, Mesh, PlaneGeometry, ShaderMaterial, UniformsLib, UniformsUtils, Vector3 } from 'three';
import { Config } from '../core/Config';
import type { V3 } from '../core/entities';
import type { Events } from '../core/Events';
import type { Rect } from '../data/districts';
import { Groups, type Physics } from '../physics/Physics';
import { isOnLand } from './city/Ground';
import type { StaticEntities } from './city/StaticEntities';

const W = Config.world;

/** Swamp pools in Grassrivers (shallow, murky). */
export const SWAMP_POOLS: Rect[] = [
  { x0: -990, z0: -820, x1: -880, z1: -690 },
  { x0: -980, z0: -540, x1: -840, z1: -420 },
  { x0: -990, z0: -120, x1: -920, z1: 40 },
  { x0: -880, z0: -60, x1: -815, z1: 120 },
  { x0: -985, z0: 380, x1: -860, z1: 520 },
  { x0: -960, z0: 640, x1: -830, z1: 840 },
];

/** Sensor volumes covering every water region (world-space rects). */
const WATER_VOLUMES: Rect[] = [
  { x0: 350, z0: -1300, x1: 500, z1: 1300 },
  { x0: 500, z0: 650, x1: 1300, z1: 1300 },
  { x0: 500, z0: -1300, x1: 1300, z1: -650 },
  { x0: 940, z0: -650, x1: 1300, z1: 650 },
  { x0: -1300, z0: -1300, x1: 350, z1: -900 },
  { x0: -1300, z0: 900, x1: 350, z1: 1300 },
  { x0: -1300, z0: -1300, x1: -1000, z1: 1300 },
];

const vertexShader = /* glsl */ `
  #include <fog_pars_vertex>
  varying vec3 vWorldPos;
  void main() {
    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
    vWorldPos = worldPosition.xyz;
    vec4 mvPosition = viewMatrix * worldPosition;
    gl_Position = projectionMatrix * mvPosition;
    #include <fog_vertex>
  }
`;

const fragmentShader = /* glsl */ `
  #include <fog_pars_fragment>
  uniform float uTime;
  uniform vec3 uDeep;
  uniform vec3 uShallow;
  uniform vec3 uSky;
  uniform vec3 uSunDir;
  uniform vec3 uSunColor;
  uniform float uOpacity;
  varying vec3 vWorldPos;

  float wave(vec2 p, vec2 dir, float freq, float speed) {
    return sin(dot(p, dir) * freq + uTime * speed);
  }

  void main() {
    vec2 p = vWorldPos.xz;
    float h = wave(p, vec2(0.8, 0.6), 0.35, 1.1) * 0.5 + wave(p, vec2(-0.5, 0.9), 0.7, 1.7) * 0.3 + wave(p, vec2(0.3, -1.0), 1.6, 2.3) * 0.2;
    // Analytic-ish normal from wave slopes.
    vec3 n = normalize(vec3(
      0.35 * 0.8 * cos(dot(p, vec2(0.8, 0.6)) * 0.35 + uTime * 1.1) * 0.5 + 0.7 * -0.5 * cos(dot(p, vec2(-0.5, 0.9)) * 0.7 + uTime * 1.7) * 0.3,
      1.0,
      0.35 * 0.6 * cos(dot(p, vec2(0.8, 0.6)) * 0.35 + uTime * 1.1) * 0.5 + 0.7 * 0.9 * cos(dot(p, vec2(-0.5, 0.9)) * 0.7 + uTime * 1.7) * 0.3
    ));
    vec3 viewDir = normalize(cameraPosition - vWorldPos);
    float fresnel = pow(1.0 - max(dot(n, viewDir), 0.0), 3.0);
    vec3 base = mix(uDeep, uShallow, 0.5 + 0.5 * h);
    vec3 color = mix(base, uSky, 0.15 + 0.5 * fresnel);
    vec3 halfDir = normalize(uSunDir + viewDir);
    float spec = pow(max(dot(n, halfDir), 0.0), 180.0);
    color += uSunColor * spec * 1.6;
    gl_FragColor = vec4(color, uOpacity);
    #include <fog_fragment>
  }
`;

/**
 * Ocean/bay/swamp water: animated shader planes, sensor trigger volumes that emit
 * `water:entered` / `water:exited` for registered entities, and `isInWater(pos)`.
 */
export class Water {
  readonly group = new Group();
  private readonly ocean: ShaderMaterial;
  private readonly swamp: ShaderMaterial;
  private readonly materials: ShaderMaterial[];

  constructor(
    physics: Physics,
    private readonly events: Events,
    entities: StaticEntities,
    parent: Group,
  ) {
    this.group.name = 'water';
    this.ocean = this.makeMaterial(new Color(0x0b3d5c), new Color(0x1f8fb0), 0.94);
    this.swamp = this.makeMaterial(new Color(0x2f4a2a), new Color(0x5a7a3a), 0.9);
    this.materials = [this.ocean, this.swamp];

    const size = W.size * 2.6;
    const oceanMesh = new Mesh(new PlaneGeometry(size, size, 1, 1), this.ocean);
    oceanMesh.rotation.x = -Math.PI / 2;
    oceanMesh.position.y = W.waterLevel;
    oceanMesh.name = 'ocean';
    oceanMesh.matrixAutoUpdate = false;
    oceanMesh.updateMatrix();
    this.group.add(oceanMesh);

    for (const pool of SWAMP_POOLS) {
      const mesh = new Mesh(new PlaneGeometry(pool.x1 - pool.x0, pool.z1 - pool.z0, 1, 1), this.swamp);
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.set((pool.x0 + pool.x1) / 2, 0.06, (pool.z0 + pool.z1) / 2);
      mesh.matrixAutoUpdate = false;
      mesh.updateMatrix();
      this.group.add(mesh);
    }
    parent.add(this.group);

    for (const r of [...WATER_VOLUMES, ...SWAMP_POOLS]) {
      const isPool = SWAMP_POOLS.includes(r);
      physics.addStaticCuboid(
        { x: (r.x0 + r.x1) / 2, y: isPool ? -0.3 : -1.5, z: (r.z0 + r.z1) / 2 },
        { x: (r.x1 - r.x0) / 2, y: isPool ? 0.7 : 2.2, z: (r.z1 - r.z0) / 2 },
        { entity: entities.water, groups: Groups.TRIGGER, sensor: true },
      );
    }
    this.events.on('physics:contactStarted', (e) => {
      if (e.a === entities.water && e.b) this.events.emit('water:entered', { entity: e.b });
      else if (e.b === entities.water && e.a) this.events.emit('water:entered', { entity: e.a });
    });
    this.events.on('physics:contactEnded', (e) => {
      if (e.a === entities.water && e.b) this.events.emit('water:exited', { entity: e.b });
      else if (e.b === entities.water && e.a) this.events.emit('water:exited', { entity: e.a });
    });
  }

  private makeMaterial(deep: Color, shallow: Color, opacity: number): ShaderMaterial {
    return new ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: UniformsUtils.merge([
        UniformsLib.fog,
        {
          uTime: { value: 0 },
          uDeep: { value: deep },
          uShallow: { value: shallow },
          uSky: { value: new Color(0.6, 0.7, 0.9) },
          uSunDir: { value: new Vector3(0, 1, 0) },
          uSunColor: { value: new Color(1, 0.9, 0.7) },
          uOpacity: { value: opacity },
        },
      ]),
      transparent: true,
      depthWrite: false,
      side: DoubleSide,
      fog: true,
    });
  }

  /** True for any point below the water surface that is not on land (or inside a swamp pool). */
  isInWater(pos: V3): boolean {
    if (pos.y > W.waterLevel + 0.5) return false;
    if (!isOnLand(pos.x, pos.z)) return true;
    for (const p of SWAMP_POOLS) if (pos.x >= p.x0 && pos.x <= p.x1 && pos.z >= p.z0 && pos.z <= p.z1) return true;
    return false;
  }

  update(time: number, skyColor: Color, sunDir: Vector3, sunColor: Color, night01: number): void {
    for (const m of this.materials) {
      const u = m.uniforms;
      u.uTime.value = time;
      (u.uSky.value as Color).copy(skyColor).multiplyScalar(1 - 0.6 * night01);
      (u.uSunDir.value as Vector3).copy(sunDir);
      (u.uSunColor.value as Color).copy(sunColor).multiplyScalar(1 - 0.7 * night01);
    }
  }

  dispose(): void {
    this.group.parent?.remove(this.group);
    this.group.traverse((o) => {
      const m = o as Mesh;
      if (m.isMesh) m.geometry.dispose();
    });
    for (const m of this.materials) m.dispose();
  }
}
