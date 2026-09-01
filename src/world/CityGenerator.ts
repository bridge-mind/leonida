import { Group, type Mesh } from 'three';
import { Config } from '../core/Config';
import type { Game } from '../core/Game';
import { Random } from '../core/Random';
import { CityMaterials } from '../gfx/Materials';
import { signAtlas, type SignAtlas } from '../gfx/Textures';
import { buildGround } from './city/Ground';
import { ChunkBatches } from './city/ChunkBatches';
import { createStaticEntities, type StaticEntities } from './city/StaticEntities';
import type { Districts } from './Districts';
import type { Landmarks } from './Landmarks';
import { RoadGraph } from './RoadGraph';
import { buildRoadGraph } from './RoadGraphBuilder';
import { ROAD_LINES } from './roadNetwork';
import { Water } from './Water';

export interface CityBuild {
  roadGraph: RoadGraph;
  materials: CityMaterials;
  entities: StaticEntities;
  water: Water;
  signs: SignAtlas;
  root: Group;
  meshes: Mesh[];
  stats: { buildings: number; triangles: number; meshes: number; chunks: number; colliders: number };
}

const SIGN_COLORS: Record<string, number> = {
  casino: 0xffd23f,
  gunstore: 0xff8a1f,
  store: 0x35e6ff,
  gas: 0xff4d4d,
  food: 0xffb347,
  safehouse: 0x4ee06a,
  bank: 0xd9c9a3,
  jewelry: 0xff4fa3,
  club: 0xff4fa3,
  respray: 0xa35bff,
  chopshop: 0xffd23f,
  marina: 0x35e6ff,
  police: 0x6fa8ff,
  hospital: 0xff5c5c,
  prison: 0xd0d0d0,
};

/**
 * Seeded, deterministic generation of the whole city. Everything static is merged
 * into per-chunk meshes; colliders are created directly in the physics world.
 */
export function generateCity(game: Game, districts: Districts, landmarks: Landmarks): CityBuild {
  const rng = Random.fromString(Config.seed);
  const root = new Group();
  root.name = 'city';
  game.worldRoot.add(root);

  const signs = signAtlas(
    landmarks.list.map((l) => ({ id: l.id, text: l.name.replace(/\s*\(.*\)/, ''), color: SIGN_COLORS[l.kind] ?? 0xffffff })),
  );
  const materials = new CityMaterials(Config.seed, signs.texture);
  const entities = createStaticEntities();
  const collidersBefore = game.physics.world.colliders.len();

  const roadGraph = buildRoadGraph(ROAD_LINES);
  const ground = buildGround({ physics: game.physics, materials, districts, entities, parent: root });
  const water = new Water(game.physics, game.events, entities, root);
  const batches = new ChunkBatches();
  rng.next();

  const built = batches.build(root, materials);
  const meshes = [...ground.meshes, ...built.meshes];
  return {
    roadGraph,
    materials,
    entities,
    water,
    signs,
    root,
    meshes,
    stats: {
      buildings: 0,
      triangles: built.triangles,
      meshes: meshes.length + water.group.children.length,
      chunks: batches.chunkCount,
      colliders: game.physics.world.colliders.len() - collidersBefore,
    },
  };
}
