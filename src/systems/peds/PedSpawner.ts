import { Frustum, Matrix4, Vector3 } from 'three';
import type { Game } from '../../core/Game';
import type { Random } from '../../core/Random';
import type { DistrictDef } from '../../data/districts';
import { PED_TYPE_IDS, PED_TYPES, type PedTypeId } from '../../data/peds';
import type { Ped } from '../../entities/Ped';
import type { SidewalkPoint } from '../../world/RoadGraph';
import { CivilianBrain } from './brains/CivilianBrain';
import { PedConfig } from './config';
import type { PedSystem } from './index';

const S = PedConfig.spawn;

/**
 * Keeps the ambient crowd around the player: target count from district density and
 * time of day, spawns on sidewalk points outside the camera frustum, despawns beyond
 * `despawnRadius`. Some spawns land as 2–3 person idle groups.
 */
export class PedSpawner {
  /** Debug override for the target count (null = computed). */
  densityOverride: number | null = null;
  private timer = 0;
  private readonly frustum = new Frustum();
  private readonly projView = new Matrix4();
  private readonly probe = new Vector3();
  private readonly spawnPos = new Vector3();

  constructor(
    private readonly game: Game,
    private readonly system: PedSystem,
    private readonly rng: Random,
  ) {}

  targetCount(playerPos: Vector3): number {
    if (this.densityOverride !== null) return this.densityOverride;
    const district = this.game.world.districtAt(playerPos);
    const density = district ? district.pedDensity : S.fallbackDensity;
    const night = this.game.time.isNight;
    const timeFactor = !night ? 1 : district?.id === 'viceBeach' ? S.beachNightFactor : S.nightFactor;
    return Math.max(S.minCount, Math.min(S.maxCount, Math.round(S.baseCount * density * timeFactor)));
  }

  update(dt: number, playerPos: Vector3): void {
    this.timer -= dt;
    if (this.timer > 0) return;
    this.timer = S.interval;
    // Interiors sit far above the city at the same x/z; do not populate the street under them.
    if (this.game.state.interior !== null) return;

    const live = this.system.live;
    let near = 0;
    for (let i = live.length - 1; i >= 0; i--) {
      const ped = live[i];
      const d = horizontal(ped.position, playerPos);
      if (d > S.despawnRadius && ped.state !== 'scripted') {
        this.system.despawn(ped);
        continue;
      }
      if (d <= S.keepRadius && ped.isAlive) near++;
    }

    let deficit = this.targetCount(playerPos) - near;
    if (deficit <= 0) return;
    const cam = this.game.viewCamera;
    this.projView.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.projView);
    const graph = this.game.world.roadGraph;
    const night = this.game.time.isNight;

    let spawned = 0;
    let rejected = 0;
    while (deficit > 0 && spawned < S.perTick && rejected < S.frustumRetries) {
      const pt = graph.randomSidewalkPointNear(playerPos, S.minRadius, S.maxRadius, this.rng);
      if (!pt) return;
      this.probe.set(pt.point.x, pt.point.y + 1, pt.point.z);
      if (this.frustum.containsPoint(this.probe)) {
        rejected++;
        continue;
      }
      const district = this.game.world.districtAt(pt.point);
      const typeId = this.pickType(district, night);
      if (deficit >= 2 && this.rng.chance(S.groupChance)) {
        const n = this.rng.int(2, 3);
        this.spawnGroup(pt, typeId, n, night, district);
        deficit -= n;
        spawned += n;
      } else {
        const heading = Math.atan2(pt.direction.x, pt.direction.z) + (this.rng.chance(0.5) ? Math.PI : 0);
        this.system.spawn(typeId, pt.point, { heading });
        deficit--;
        spawned++;
      }
    }
  }

  /** Weighted type pick for a district; falls back to tourists where nothing is listed. */
  pickType(district: DistrictDef | null, night: boolean): PedTypeId {
    let total = 0;
    for (let i = 0; i < PED_TYPE_IDS.length; i++) total += this.weight(PED_TYPES[PED_TYPE_IDS[i]], district, night);
    if (total <= 0) return 'tourist';
    let roll = this.rng.next() * total;
    for (let i = 0; i < PED_TYPE_IDS.length; i++) {
      const id = PED_TYPE_IDS[i];
      roll -= this.weight(PED_TYPES[id], district, night);
      if (roll <= 0) return id;
    }
    return 'tourist';
  }

  private weight(type: (typeof PED_TYPES)[PedTypeId], district: DistrictDef | null, night: boolean): number {
    if (!type.ambient || !district) return 0;
    const w = type.districtWeights[district.id] ?? 0;
    return w * (night ? type.nightWeight : type.dayWeight);
  }

  private spawnGroup(pt: SidewalkPoint, typeId: PedTypeId, count: number, night: boolean, district: DistrictDef | null): void {
    const center = pt.point;
    const seconds = this.rng.range(PedConfig.idleGroup.durationMin, PedConfig.idleGroup.durationMax);
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2 + this.rng.range(-0.3, 0.3);
      this.spawnPos.set(center.x + Math.cos(angle) * PedConfig.idleGroup.radius, center.y, center.z + Math.sin(angle) * PedConfig.idleGroup.radius);
      // Groups mix types a little so they do not look like uniforms.
      const id = i === 0 || this.rng.chance(0.6) ? typeId : this.pickType(district, night);
      const ped = this.system.spawn(id, this.spawnPos, { heading: Math.atan2(center.x - this.spawnPos.x, center.z - this.spawnPos.z) }) as Ped;
      if (ped.brain instanceof CivilianBrain) ped.brain.enterIdleGroup(ped, center, seconds);
    }
  }
}

function horizontal(a: Vector3, b: Vector3): number {
  const dx = a.x - b.x;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dz * dz);
}
