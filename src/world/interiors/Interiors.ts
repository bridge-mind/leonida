import { Color, CylinderGeometry, Group, InstancedMesh, Matrix4, MeshBasicMaterial, Object3D, Vector3 } from 'three';
import { Config } from '../../core/Config';
import type { V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { BlipKind } from '../../core/GameState';
import type { LandmarkDef } from '../../data/landmarks';
import type { Physics } from '../../physics/Physics';

export interface InteriorHandle {
  root: Object3D;
  dispose(): void;
  /** Called every frame while the interior is active. */
  update?(dt: number): void;
  /** Called when the player enters / leaves (guards, clerks, music). */
  onEnter?(): void;
  onExit?(): void;
}

export interface InteriorBuildContext {
  /** Group already positioned at the interior origin; add meshes here in local coordinates. */
  root: Group;
  physics: Physics;
  game: Game;
  /** World position of the interior origin (`Config.interiors.origin`). */
  origin: Vector3;
  /** Converts a local point to world space (for colliders, which are always world-space). */
  toWorld(local: V3): Vector3;
}

export interface InteriorDefinition {
  id: string;
  build(ctx: InteriorBuildContext): InteriorHandle;
  /** Local position where the player appears on entering. */
  spawnPoint: V3;
  /** Rig heading on entering (0 faces +Z). */
  spawnHeading?: number;
  /** Local point that acts as the exit door (default: spawnPoint). */
  exitTrigger?: V3;
  /** World exit position; defaults to the matching landmark door. */
  exitPoint?: V3;
  exitHeading?: number;
}

const MARKER_COLORS: Partial<Record<BlipKind, number>> = {
  casino: 0xffd23f,
  gunstore: 0xff8a1f,
  store: 0x35e6ff,
  gas: 0x35e6ff,
  food: 0xffb347,
  safehouse: 0x4ee06a,
  bank: 0xa35bff,
  jewelry: 0xff4fa3,
  club: 0xff4fa3,
};

/**
 * Interiors framework: registration, door markers generated from landmarks, fade
 * transitions, enter/exit with `interior:entered` / `interior:exited` events.
 * Interior scenes are built lazily at `Config.interiors.origin`, far above the city.
 */
export class Interiors {
  readonly origin = new Vector3(Config.interiors.origin.x, Config.interiors.origin.y, Config.interiors.origin.z);
  private readonly definitions = new Map<string, InteriorDefinition>();
  private readonly handles = new Map<string, InteriorHandle>();
  private readonly roots = new Map<string, Group>();
  current: string | null = null;
  transitioning = false;
  private markers: InstancedMesh | null = null;
  private readonly markerLandmarks: LandmarkDef[] = [];
  private readonly fadeEl: HTMLElement | null;
  private readonly tmp = new Vector3();

  constructor(private readonly game: Game) {
    this.fadeEl = document.getElementById('fade');
  }

  /** Builds door markers for every landmark that has an interior id. */
  buildDoorMarkers(landmarks: readonly LandmarkDef[]): void {
    const withDoors = landmarks.filter((l) => l.interiorId !== null);
    this.markerLandmarks.length = 0;
    this.markerLandmarks.push(...withDoors);
    if (this.markers) {
      this.game.worldRoot.remove(this.markers);
      this.markers.dispose();
    }
    const geom = new CylinderGeometry(Config.interiors.doorRadius, Config.interiors.doorRadius, 0.06, 24, 1, true);
    const mat = new MeshBasicMaterial({ vertexColors: false, color: 0xffffff, transparent: true, opacity: 0.85 });
    const mesh = new InstancedMesh(geom, mat, Math.max(1, withDoors.length));
    const m = new Matrix4();
    const color = new Color();
    withDoors.forEach((l, i) => {
      const y = this.game.physics.groundHeightAt(l.door.x, l.door.z) ?? 0;
      m.makeTranslation(l.door.x, y + 0.5, l.door.z);
      mesh.setMatrixAt(i, m);
      color.set(MARKER_COLORS[l.kind] ?? 0xffffff);
      mesh.setColorAt(i, color);
    });
    mesh.count = withDoors.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.name = 'doorMarkers';
    mesh.frustumCulled = false;
    this.markers = mesh;
    this.game.worldRoot.add(mesh);
  }

  register(def: InteriorDefinition): void {
    this.definitions.set(def.id, def);
  }

  has(id: string): boolean {
    return this.definitions.has(id);
  }

  get registeredIds(): string[] {
    return [...this.definitions.keys()];
  }

  landmarkFor(interiorId: string): LandmarkDef | undefined {
    return this.markerLandmarks.find((l) => l.interiorId === interiorId);
  }

  /** Nearest door marker within `radius` of a world position. */
  doorNear(pos: V3, radius: number): LandmarkDef | null {
    let best: LandmarkDef | null = null;
    let bestD = radius * radius;
    for (const l of this.markerLandmarks) {
      const dx = l.door.x - pos.x;
      const dz = l.door.z - pos.z;
      const d = dx * dx + dz * dz;
      if (d < bestD) {
        bestD = d;
        best = l;
      }
    }
    return best;
  }

  private getHandle(def: InteriorDefinition): InteriorHandle {
    let handle = this.handles.get(def.id);
    if (handle) return handle;
    const root = new Group();
    root.name = `interior:${def.id}`;
    root.position.copy(this.origin);
    root.visible = false;
    this.game.interiorRoot.add(root);
    const ctx: InteriorBuildContext = {
      root,
      physics: this.game.physics,
      game: this.game,
      origin: this.origin,
      toWorld: (local) => new Vector3(local.x, local.y, local.z).add(this.origin),
    };
    handle = def.build(ctx);
    this.handles.set(def.id, handle);
    this.roots.set(def.id, root);
    return handle;
  }

  async enter(id: string): Promise<boolean> {
    const def = this.definitions.get(id);
    if (!def || this.transitioning || this.current === id) return false;
    this.transitioning = true;
    await this.fade(true);
    if (this.current) this.deactivate(this.current);
    const handle = this.getHandle(def);
    const root = this.roots.get(id) as Group;
    root.visible = true;
    this.game.worldRoot.visible = false;
    this.current = id;
    this.game.state.interior = id;
    if (this.game.systems.has('player')) {
      this.tmp.set(def.spawnPoint.x, def.spawnPoint.y, def.spawnPoint.z).add(this.origin);
      this.game.player.teleport(this.tmp, def.spawnHeading ?? 0, false);
      if (this.game.systems.has('camera')) {
        this.game.camera.setOrbit((def.spawnHeading ?? 0) + Math.PI, 0.15);
        this.game.camera.snap();
      }
    }
    handle.onEnter?.();
    // `interior.enter/exit` cues are played by the audio system from these events.
    this.game.events.emit('interior:entered', { id });
    await this.fade(false);
    this.transitioning = false;
    return true;
  }

  async exit(): Promise<boolean> {
    const id = this.current;
    if (!id || this.transitioning) return false;
    const def = this.definitions.get(id);
    if (!def) return false;
    this.transitioning = true;
    await this.fade(true);
    this.deactivate(id);
    this.game.worldRoot.visible = true;
    this.current = null;
    this.game.state.interior = null;
    const landmark = this.landmarkFor(id);
    const exitPoint = def.exitPoint ?? landmark?.door ?? { x: 0, y: 0, z: 0 };
    const heading = def.exitHeading ?? landmark?.doorHeading ?? 0;
    if (this.game.systems.has('player')) {
      // Step one meter out of the doorway so the marker does not immediately re-trigger.
      this.tmp.set(exitPoint.x + Math.sin(heading) * 1.2, exitPoint.y, exitPoint.z + Math.cos(heading) * 1.2);
      this.game.player.teleport(this.tmp, heading, true);
      if (this.game.systems.has('camera')) {
        this.game.camera.setOrbit(heading + Math.PI, 0.18);
        this.game.camera.snap();
      }
    }
    this.game.events.emit('interior:exited', { id });
    await this.fade(false);
    this.transitioning = false;
    return true;
  }

  private deactivate(id: string): void {
    const root = this.roots.get(id);
    if (root) root.visible = false;
    this.handles.get(id)?.onExit?.();
  }

  update(dt: number): void {
    if (this.current) this.handles.get(this.current)?.update?.(dt);
  }

  /** Local-space exit trigger of the current interior (for the prompt/E key). */
  currentExitTrigger(): Vector3 | null {
    if (!this.current) return null;
    const def = this.definitions.get(this.current);
    if (!def) return null;
    const p = def.exitTrigger ?? def.spawnPoint;
    return this.tmp.set(p.x, p.y, p.z).add(this.origin);
  }

  private fade(toBlack: boolean): Promise<void> {
    const el = this.fadeEl;
    if (!el) return Promise.resolve();
    el.classList.toggle('visible', toBlack);
    return new Promise((resolve) => setTimeout(resolve, Config.interiors.fadeSeconds * 1000));
  }

  dispose(): void {
    for (const h of this.handles.values()) h.dispose();
    this.handles.clear();
    if (this.markers) {
      this.game.worldRoot.remove(this.markers);
      this.markers.dispose();
    }
  }
}
