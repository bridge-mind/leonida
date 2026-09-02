import { CylinderGeometry, DoubleSide, Mesh, MeshBasicMaterial, type Group } from 'three';
import type { V3 } from '../../core/entities';
import type { INavigationPeer } from '../../core/peers';
import { MissionsConfig } from './config';

const M = MissionsConfig.marker;

interface Marker {
  id: string;
  position: V3;
  mesh: Mesh;
}

/**
 * Translucent yellow cylinders at mission contacts plus their `contact` blips. One shared
 * geometry/material; the pulse animates opacity on the shared material so all markers breathe together.
 */
export class ContactMarkers {
  private readonly markers = new Map<string, Marker>();
  private readonly geometry = new CylinderGeometry(M.radius, M.radius, M.height, 24, 1, true);
  private readonly material = new MeshBasicMaterial({ color: M.color, transparent: true, opacity: M.opacity, side: DoubleSide, depthWrite: false });
  private phase = 0;

  constructor(
    private readonly root: Group,
    private readonly navigation: () => INavigationPeer | undefined,
  ) {}

  has(id: string): boolean {
    return this.markers.has(id);
  }

  add(id: string, position: V3, label: string, groundY: number): void {
    if (this.markers.has(id)) return;
    const mesh = new Mesh(this.geometry, this.material);
    mesh.position.set(position.x, groundY + M.height / 2, position.z);
    mesh.name = `mission-marker-${id}`;
    this.root.add(mesh);
    this.markers.set(id, { id, position: { x: position.x, y: groundY, z: position.z }, mesh });
    this.navigation()?.addBlip({ id: `mission-${id}`, kind: 'contact', position: { x: position.x, y: groundY, z: position.z }, label });
  }

  remove(id: string): void {
    const m = this.markers.get(id);
    if (!m) return;
    this.markers.delete(id);
    this.root.remove(m.mesh);
    this.navigation()?.removeBlip(`mission-${id}`);
  }

  /** Marker id whose cylinder contains the horizontal point, if any. */
  at(pos: V3): string | null {
    for (const m of this.markers.values()) {
      const dx = m.position.x - pos.x;
      const dz = m.position.z - pos.z;
      if (dx * dx + dz * dz <= M.radius * M.radius) return m.id;
    }
    return null;
  }

  update(realDt: number): void {
    if (this.markers.size === 0) return;
    this.phase += realDt / M.pulseSeconds;
    this.material.opacity = M.opacity * (0.75 + 0.25 * Math.sin(this.phase * Math.PI * 2));
  }

  clear(): void {
    for (const id of Array.from(this.markers.keys())) this.remove(id);
  }

  dispose(): void {
    this.clear();
    this.geometry.dispose();
    this.material.dispose();
  }
}
