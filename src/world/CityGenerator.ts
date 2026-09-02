import { Group, type Mesh } from 'three';
import { Config } from '../core/Config';
import type { Game } from '../core/Game';
import { Random } from '../core/Random';
import { CityMaterials } from '../gfx/Materials';
import { signAtlas, type SignAtlas } from '../gfx/Textures';
import { fillBlocks } from './city/Blocks';
import { ChunkBatches } from './city/ChunkBatches';
import { buildGround } from './city/Ground';
import { buildLandmarks } from './city/LandmarkBuilder';
import { buildRoads } from './city/RoadMeshes';
import { buildSpecialAreas } from './city/SpecialAreas';
import { createStaticEntities, type StaticEntities } from './city/StaticEntities';
import type { Districts } from './Districts';
import type { Landmarks } from './Landmarks';
import { Props } from './Props';
import { RoadGraph } from './RoadGraph';
import { buildRoadGraph } from './RoadGraphBuilder';
import { ROAD_LINES } from './roadNetwork';
import { Water } from './Water';

export interface CityBuild {
  roadGraph: RoadGraph;
  materials: CityMaterials;
  entities: StaticEntities;
  water: Water;
  props: Props;
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
  const physics = game.physics;
  const collidersBefore = physics.world.colliders.len();

  const roadGraph = buildRoadGraph(ROAD_LINES);
  const ground = buildGround({ physics, materials, districts, entities, parent: root });
  const water = new Water(physics, game.events, entities, root);
  const batches = new ChunkBatches();
  const props = new Props(materials, physics, entities, rng.fork('props'));

  const roads = buildRoads({ graph: roadGraph, batches, materials, physics, entities, parent: root, props });
  const blocks = fillBlocks({ graph: roadGraph, batches, districts, landmarks, physics, entities, rng: rng.fork('blocks'), props, materials });
  const marks = buildLandmarks({ landmarks, batches, physics, entities, signs, materials, parent: root, props, rng: rng.fork('landmarks') });
  const special = buildSpecialAreas({ batches, physics, entities, props, rng: rng.fork('special'), materials, parent: root, graph: roadGraph });

  const built = batches.build(root, materials);
  const propMeshes = props.build(root);
  const meshes = [...ground.meshes, ...roads.meshes, ...built.meshes, ...propMeshes, ...special.meshes, marks.signMesh];
  return {
    roadGraph,
    materials,
    entities,
    water,
    props,
    signs,
    root,
    meshes,
    stats: {
      buildings: blocks.buildings + special.buildings + landmarks.list.length,
      triangles: built.triangles + roads.triangles,
      meshes: meshes.length + water.group.children.length,
      chunks: batches.chunkCount,
      colliders: physics.world.colliders.len() - collidersBefore,
    },
  };
}
