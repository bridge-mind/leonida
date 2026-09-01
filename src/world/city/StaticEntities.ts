import { Vector3, type Object3D } from 'three';
import { allocEntityId, type Entity, type EntityKind } from '../../core/entities';

export type StaticSurface = 'concrete' | 'asphalt' | 'sand' | 'grass' | 'dirt' | 'wood' | 'metal' | 'water' | 'glass';

/**
 * Shared entities for static geometry so raycasts and contacts resolve to something
 * meaningful (`hit.entity.kind === 'building'`, `surface === 'concrete'`).
 */
export class StaticEntity implements Entity {
  readonly id = allocEntityId();
  readonly position = new Vector3();
  isAlive = true;
  object3d: Object3D | null = null;

  constructor(
    readonly kind: EntityKind,
    readonly surface: StaticSurface,
    readonly label: string,
  ) {}
}

export interface StaticEntities {
  ground: StaticEntity;
  road: StaticEntity;
  sidewalk: StaticEntity;
  building: StaticEntity;
  glassBuilding: StaticEntity;
  prop: StaticEntity;
  metalProp: StaticEntity;
  wood: StaticEntity;
  sand: StaticEntity;
  dirt: StaticEntity;
  seabed: StaticEntity;
  water: StaticEntity;
  bridge: StaticEntity;
}

export function createStaticEntities(): StaticEntities {
  return {
    ground: new StaticEntity('ground', 'grass', 'ground'),
    road: new StaticEntity('ground', 'asphalt', 'road'),
    sidewalk: new StaticEntity('ground', 'concrete', 'sidewalk'),
    building: new StaticEntity('building', 'concrete', 'building'),
    glassBuilding: new StaticEntity('building', 'glass', 'glass building'),
    prop: new StaticEntity('prop', 'concrete', 'prop'),
    metalProp: new StaticEntity('prop', 'metal', 'metal prop'),
    wood: new StaticEntity('prop', 'wood', 'wooden structure'),
    sand: new StaticEntity('ground', 'sand', 'beach'),
    dirt: new StaticEntity('ground', 'dirt', 'swamp ground'),
    seabed: new StaticEntity('ground', 'sand', 'seabed'),
    water: new StaticEntity('water', 'water', 'water'),
    bridge: new StaticEntity('ground', 'asphalt', 'bridge'),
  };
}

export function isStaticEntity(e: Entity | null): e is StaticEntity {
  return e instanceof StaticEntity;
}
