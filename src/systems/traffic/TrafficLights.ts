import type { Props } from '../../world/Props';
import type { RoadEdge, RoadGraph, RoadNode } from '../../world/RoadGraph';
import { TrafficConfig } from './config';

export type SignalAxis = 'ns' | 'ew';

export interface SignalState {
  /** Axis with the green (or yellow) light; null during the all-red gap. */
  green: SignalAxis | null;
  yellow: boolean;
  /** Seconds until the current green turns red; Infinity while all-red. */
  timeToRed: number;
}

const L = TrafficConfig.lights;

/**
 * Pure light clock: each lit node cycles `axis0` green (12 s) → all red (2 s) →
 * `axis1` green → all red. `lightPhase` from the road graph decides which axis goes
 * first so neighbouring intersections are staggered; the world builds its lamps with
 * the same rule (phase 0 = N–S green at t = 0).
 */
export function signalAt(t: number, lightPhase: number, out: SignalState): SignalState {
  const half = L.greenSeconds + L.allRedSeconds;
  const cycle = half * 2;
  const u = ((t % cycle) + cycle) % cycle;
  const axis0: SignalAxis = lightPhase === 0 ? 'ns' : 'ew';
  const axis1: SignalAxis = axis0 === 'ns' ? 'ew' : 'ns';
  if (u < L.greenSeconds) {
    out.green = axis0;
    out.timeToRed = L.greenSeconds - u;
  } else if (u < half) {
    out.green = null;
    out.timeToRed = Infinity;
  } else if (u < half + L.greenSeconds) {
    out.green = axis1;
    out.timeToRed = half + L.greenSeconds - u;
  } else {
    out.green = null;
    out.timeToRed = Infinity;
  }
  out.yellow = out.green !== null && out.timeToRed < L.yellowSeconds;
  return out;
}

export function edgeAxis(edge: RoadEdge): SignalAxis {
  return Math.abs(edge.direction.z) > Math.abs(edge.direction.x) ? 'ns' : 'ew';
}

/** Drives the clock for every lit node and mirrors it onto the world's signal lamps. */
export class TrafficLights {
  time = 0;
  private readonly nodes: RoadNode[];
  private readonly lastCode: Int8Array;
  private readonly state: SignalState = { green: null, yellow: false, timeToRed: Infinity };

  constructor(
    private readonly graph: RoadGraph,
    private readonly props: Props | null,
  ) {
    this.nodes = graph.nodes.filter((n) => n.hasLight);
    this.lastCode = new Int8Array(graph.nodes.length).fill(-1);
  }

  get lightNodes(): readonly RoadNode[] {
    return this.nodes;
  }

  update(dt: number): void {
    this.time += dt;
    const s = this.state;
    for (let i = 0; i < this.nodes.length; i++) {
      const node = this.nodes[i];
      signalAt(this.time, node.lightPhase, s);
      const code = (s.green === null ? 0 : s.green === 'ns' ? 1 : 2) + (s.yellow ? 3 : 0);
      if (this.lastCode[node.id] === code) continue;
      this.lastCode[node.id] = code;
      this.props?.setSignal(node.id, s.green === null ? null : s.green === 'ns', s.yellow ? s.green : null);
    }
  }

  /** Fills `out` with the light state for travelling through `nodeId` along `edgeId`. */
  stateFor(nodeId: number, edgeId: number, out: SignalState): SignalState {
    const node = this.graph.nodes[nodeId];
    if (!node.hasLight) {
      out.green = edgeAxis(this.graph.edges[edgeId]);
      out.yellow = false;
      out.timeToRed = Infinity;
      return out;
    }
    signalAt(this.time, node.lightPhase, out);
    return out;
  }

  isGreen(nodeId: number, edgeId: number): boolean {
    const node = this.graph.nodes[nodeId];
    const edge = this.graph.edges[edgeId];
    if (!node || !edge) return true;
    if (!node.hasLight) return true;
    signalAt(this.time, node.lightPhase, this.state);
    return this.state.green === edgeAxis(edge);
  }
}
