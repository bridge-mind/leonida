import { Color, Vector3 } from 'three';
import type { V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import type { DistrictDef } from '../../data/districts';
import { generateCity, type CityBuild } from '../../world/CityGenerator';
import { DayNight } from '../../world/DayNight';
import { Districts } from '../../world/Districts';
import { Landmarks } from '../../world/Landmarks';
import type { RoadGraph } from '../../world/RoadGraph';

/**
 * Owns the static city, the road graph, districts, landmarks, water and day/night.
 * Public API: `roadGraph`, `districts`, `landmarks`, `dayNight`, `isInWater(pos)`,
 * `groundHeightAt(x, z)`, `districtAt(pos)`, `city.entities` (static entity kinds).
 */
export class WorldSystem implements GameSystem {
  readonly districts = new Districts();
  readonly landmarks = new Landmarks();
  dayNight!: DayNight;
  city!: CityBuild;
  private readonly sunColor = new Color();
  private readonly focus = new Vector3();

  constructor(private readonly game: Game) {}

  init(): void {
    const t0 = performance.now();
    this.dayNight = new DayNight(this.game);
    this.city = generateCity(this.game, this.districts, this.landmarks);
    const ms = performance.now() - t0;
    const s = this.city.stats;
    console.info(
      `[world] generated in ${ms.toFixed(0)} ms: ${this.roadGraph.nodes.length} nodes, ${this.roadGraph.edges.length} edges, ` +
        `${s.buildings} buildings, ${s.meshes} meshes, ${(s.triangles / 1000).toFixed(0)}k tris, ${s.colliders} colliders`,
    );
    this.game.perf.set('buildings', s.buildings);
    this.game.perf.set('staticMeshes', s.meshes);
  }

  get roadGraph(): RoadGraph {
    return this.city.roadGraph;
  }

  isInWater(pos: V3): boolean {
    return this.city.water.isInWater(pos);
  }

  groundHeightAt(x: number, z: number): number {
    return this.game.physics.groundHeightAt(x, z) ?? 0;
  }

  districtAt(pos: V3): DistrictDef | null {
    return this.districts.at(pos.x, pos.z);
  }

  /** 0 = day, 1 = lamps/neon/windows fully on. */
  get night01(): number {
    return this.dayNight.night01;
  }

  update(dt: number): void {
    const player = this.game.systems.has('player') ? this.game.player : undefined;
    if (player) this.focus.copy(player.position);
    this.dayNight.update(dt, player ? this.focus : null);
    const night = this.dayNight.night01;
    this.city.materials.setNight(night);
    this.sunColor.copy(this.dayNight.sun.color);
    this.city.water.update(this.game.time.realElapsed, this.dayNight.horizonColor, this.dayNight.sunDirection, this.sunColor, night);
  }

  dispose(): void {
    this.city.water.dispose();
    this.city.materials.dispose();
    this.dayNight.dispose();
    this.game.worldRoot.remove(this.city.root);
  }

  debugActions = {
    /** Teleports the player to a landmark door (requires the player system). */
    goto: (landmarkId: unknown): V3 | null => {
      const l = this.landmarks.byId(String(landmarkId));
      if (!l || !this.game.systems.has('player')) return null;
      this.game.player.teleport(l.door, l.doorHeading);
      return l.door;
    },
    landmarks: (): string[] => this.landmarks.list.map((l) => l.id),
    stats: (): Record<string, unknown> => ({ ...this.city.stats, nodes: this.roadGraph.nodes.length, edges: this.roadGraph.edges.length }),
    night: (): number => this.dayNight.night01,
    inWater: (x: unknown, z: unknown): boolean => this.isInWater({ x: Number(x), y: 0, z: Number(z) }),
    ground: (x: unknown, z: unknown): number | null => this.game.physics.groundHeightAt(Number(x), Number(z)),
  };
}

const module: SystemModule = {
  id: 'world',
  priority: PRIORITY.world,
  create: (game) => new WorldSystem(game),
};

export default module;
