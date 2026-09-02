import { Config } from '../core/Config';
import { DISTRICTS, type Rect } from '../data/districts';
import type { LandmarkDef } from '../data/landmarks';
import type { RoadGraph } from '../world/RoadGraph';
import { UiConfig } from '../systems/ui/config';

/**
 * The static city map rendered once to an offscreen canvas. Both the minimap and the
 * full map blit it with their own transform, so per-frame cost is a single drawImage.
 * Layer pixel (0, 0) is world (-halfExtent, -halfExtent); +x east, +y (canvas) = +z south.
 */
export class MapLayer {
  readonly canvas: HTMLCanvasElement;
  readonly pxPerMeter: number;
  readonly halfExtent: number;
  readonly size: number;

  constructor(graph: RoadGraph, landmarks: readonly LandmarkDef[]) {
    this.pxPerMeter = UiConfig.map.staticPxPerMeter;
    this.halfExtent = UiConfig.map.halfExtent;
    this.size = Math.ceil(this.halfExtent * 2 * this.pxPerMeter);
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.size;
    this.canvas.height = this.size;
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('2d canvas unavailable for the map layer');
    this.paint(ctx, graph, landmarks);
  }

  /** Layer-space x for a world x (same for z → y). */
  toLayer(world: number): number {
    return (world + this.halfExtent) * this.pxPerMeter;
  }

  private paint(ctx: CanvasRenderingContext2D, graph: RoadGraph, landmarks: readonly LandmarkDef[]): void {
    const c = UiConfig.map.colors;
    const s = this.pxPerMeter;
    ctx.fillStyle = c.water;
    ctx.fillRect(0, 0, this.size, this.size);

    ctx.fillStyle = c.land;
    for (const r of [Config.world.mainland, Config.world.island, Config.world.swamp]) this.fillRect(ctx, r);

    for (const d of DISTRICTS) {
      ctx.fillStyle = hex(d.mapColor);
      this.fillRect(ctx, d.rect);
    }

    // Roads: outline pass then fill pass so intersections merge cleanly.
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (let pass = 0; pass < 2; pass++) {
      for (const e of graph.edges) {
        const a = graph.nodes[e.a].position;
        const b = graph.nodes[e.b].position;
        const w = Math.max(1.5, e.width * s);
        if (pass === 0) {
          ctx.strokeStyle = c.roadOutline;
          ctx.lineWidth = w + Math.max(1, s * 1.5);
        } else {
          ctx.strokeStyle =
            e.surface === 'dirt' ? c.roadDirt : e.isHighway ? c.roadHighway : e.kind === 'avenue' ? c.roadMajor : c.roadMinor;
          ctx.lineWidth = w;
        }
        ctx.beginPath();
        ctx.moveTo(this.toLayer(a.x), this.toLayer(a.z));
        ctx.lineTo(this.toLayer(b.x), this.toLayer(b.z));
        ctx.stroke();
      }
    }

    ctx.fillStyle = c.footprint;
    for (const l of landmarks) {
      const w = l.footprint.w * s;
      const d = l.footprint.d * s;
      ctx.fillRect(this.toLayer(l.position.x) - w / 2, this.toLayer(l.position.z) - d / 2, w, d);
    }
  }

  private fillRect(ctx: CanvasRenderingContext2D, r: Rect): void {
    ctx.fillRect(this.toLayer(r.x0), this.toLayer(r.z0), (r.x1 - r.x0) * this.pxPerMeter, (r.z1 - r.z0) * this.pxPerMeter);
  }
}

function hex(color: number): string {
  return `#${color.toString(16).padStart(6, '0')}`;
}
