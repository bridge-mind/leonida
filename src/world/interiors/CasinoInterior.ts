import { BoxGeometry, Group, Mesh, MeshStandardMaterial, PointLight, type Object3D } from 'three';
import { WHEEL_SEGMENT_COUNT, segmentAngle } from '../../casino/LuckyWheel';
import { pocketAngle } from '../../casino/Roulette';
import type { SlotSymbol } from '../../casino/Slots';
import type { Game } from '../../core/Game';
import { Groups } from '../../physics/Physics';
import { CasinoConfig } from '../../systems/casino/config';
import { PropKit } from '../../systems/casino/props';
import {
  blackjackFeltTexture,
  carpetTexture,
  luckyWheelTexture,
  rouletteLayoutTexture,
  rouletteWheelTexture,
  type SlotScreen,
} from '../../systems/casino/textures';
import type { InteriorBuildContext, InteriorDefinition, InteriorHandle } from './Interiors';

export type StationId = keyof typeof CasinoConfig.stations;

const STATION_PROMPTS: Record<StationId, string> = {
  cashier: 'Cashier',
  blackjackLow: `Blackjack ($${CasinoConfig.tables.low.min}–$${CasinoConfig.tables.low.max.toLocaleString('en-US')})`,
  blackjackHigh: `Blackjack ($${CasinoConfig.tables.high.min}–$${CasinoConfig.tables.high.max.toLocaleString('en-US')})`,
  roulette: 'Roulette',
  slots: 'Slots',
  wheel: 'Lucky Wheel',
};

const STATION_IDS = Object.keys(CasinoConfig.stations) as StationId[];

/** What the casino system needs from the interior while the player is inside. */
export interface CasinoInteriorHost {
  readonly game: Game;
  /** Player pressed E at a station with no UI open. */
  onStation(station: StationId): void;
  isUiOpen(): boolean;
}

/** 3D animation hooks the UI panels drive. All no-ops until the interior has been built. */
export interface CasinoStage {
  spinRoulette(pocket: number, seconds: number): void;
  spinWheel(segment: number, seconds: number, onTick?: () => void): void;
  setSlotScreen(machine: number, reels: readonly SlotSymbol[] | null, spinning: boolean): void;
  /** Index of the slot machine closest to the player. */
  nearestSlot(): number;
  readonly built: boolean;
}

interface SpinAnim {
  active: boolean;
  t: number;
  duration: number;
  from: number;
  to: number;
  lastTick: number;
  onTick: (() => void) | null;
}

function easeOutCubic(t: number): number {
  const u = 1 - t;
  return 1 - u * u * u;
}

/** Shortest forward distance so the animation always spins forward and lands exactly on `target`. */
function spinTarget(from: number, target: number, turns: number): number {
  const twoPi = Math.PI * 2;
  const delta = (((target - from) % twoPi) + twoPi) % twoPi;
  return from + delta + turns * twoPi;
}

class CasinoInteriorHandle implements InteriorHandle, CasinoStage {
  readonly root: Object3D;
  readonly built = true;
  private readonly kit: PropKit;
  private readonly rouletteWheel: Mesh;
  private readonly rouletteBall: Mesh;
  private readonly luckyDisc: Mesh;
  private readonly slotScreens: SlotScreen[] = [];
  private readonly slotLocal: { x: number; z: number }[] = [];
  private readonly roulette: SpinAnim = { active: false, t: 0, duration: 1, from: 0, to: 0, lastTick: 0, onTick: null };
  private readonly wheel: SpinAnim = { active: false, t: 0, duration: 1, from: 0, to: 0, lastTick: 0, onTick: null };
  private readonly prompt = { key: 'E', text: '' };
  private promptShown = false;
  private readonly lights: PointLight[] = [];

  constructor(
    private readonly ctx: InteriorBuildContext,
    private readonly host: CasinoInteriorHost,
  ) {
    this.root = ctx.root;
    this.kit = new PropKit(ctx);
    this.buildShell();
    const built = this.buildFurniture();
    this.rouletteWheel = built.wheel;
    this.rouletteBall = built.ball;
    this.luckyDisc = built.disc;
    this.buildLights();
  }

  private buildShell(): void {
    const kit = this.kit;
    const room = CasinoConfig.room;
    const c = room.center;
    const t = room.wallThickness;
    const carpet = carpetTexture();
    carpet.repeat.set(room.width / 4, room.depth / 4);
    const floorMat = kit.std(0xffffff, 0.95, 0, carpet);
    kit.box(this.root, room.width, 0.2, room.depth, floorMat, c.x, -0.1, c.z, true);
    const wall = kit.mat.plum;
    const hw = room.width / 2;
    const hd = room.depth / 2;
    const hy = room.height / 2;
    kit.box(this.root, room.width + t * 2, room.height, t, wall, c.x, hy, c.z - hd - t / 2, true);
    kit.box(this.root, room.width + t * 2, room.height, t, wall, c.x, hy, c.z + hd + t / 2, true);
    kit.box(this.root, t, room.height, room.depth, wall, c.x - hw - t / 2, hy, c.z, true);
    kit.box(this.root, t, room.height, room.depth, wall, c.x + hw + t / 2, hy, c.z, true);
    // Gold trim at the skirting and the cornice.
    for (const y of [0.12, room.height - 0.15]) {
      kit.box(this.root, room.width, 0.24, 0.08, kit.mat.gold, c.x, y, c.z - hd + 0.04);
      kit.box(this.root, room.width, 0.24, 0.08, kit.mat.gold, c.x, y, c.z + hd - 0.04);
      kit.box(this.root, 0.08, 0.24, room.depth, kit.mat.gold, c.x - hw + 0.04, y, c.z);
      kit.box(this.root, 0.08, 0.24, room.depth, kit.mat.gold, c.x + hw - 0.04, y, c.z);
    }
    const ceiling = new Mesh(new BoxGeometry(room.width, 0.2, room.depth), new MeshStandardMaterial({ color: 0x1a1020, roughness: 1 }));
    ceiling.position.set(c.x, room.height + 0.1, c.z);
    this.root.add(ceiling);
    kit.geo(ceiling.geometry);
    for (let i = -1; i <= 1; i++) {
      kit.lightStrip(this.root, 0.5, room.depth * 0.8, c.x + i * 12, room.height - 0.02, c.z);
    }
    // Entrance doorway accent so the exit is readable from inside.
    kit.box(this.root, 3.2, 0.3, 0.3, kit.mat.gold, 0, 3.2, c.z + hd - 0.15);
    kit.box(this.root, 0.3, 3.2, 0.3, kit.mat.gold, -1.6, 1.6, c.z + hd - 0.15);
    kit.box(this.root, 0.3, 3.2, 0.3, kit.mat.gold, 1.6, 1.6, c.z + hd - 0.15);
    // Physics floor collider that also catches the player if they clip a wall corner.
    this.ctx.physics.addStaticCuboid(this.ctx.toWorld({ x: c.x, y: -1.1, z: c.z }), { x: hw + 4, y: 1, z: hd + 4 }, { groups: Groups.STATIC });
  }

  private buildFurniture(): { wheel: Mesh; ball: Mesh; disc: Mesh } {
    const kit = this.kit;
    const S = CasinoConfig.stations;
    const at = (x: number, z: number): Group => {
      const g = new Group();
      g.position.set(x, 0, z);
      this.root.add(g);
      return g;
    };
    kit.cashierCage(at(S.cashier.x - 2.4, S.cashier.z));
    kit.blackjackTable(at(S.blackjackLow.x, S.blackjackLow.z - 2.4), blackjackFeltTexture(CasinoConfig.tables.low), CasinoConfig.tables.low);
    kit.blackjackTable(at(S.blackjackHigh.x, S.blackjackHigh.z - 2.4), blackjackFeltTexture(CasinoConfig.tables.high), CasinoConfig.tables.high);
    const rouletteGroup = at(S.roulette.x, S.roulette.z - 2.4);
    const { wheel, ball } = kit.rouletteTable(rouletteGroup, rouletteLayoutTexture(), rouletteWheelTexture());
    const slotGroup = at(S.slots.x + 2.0, S.slots.z);
    const count = CasinoConfig.slots.machineCount;
    for (let i = 0; i < count; i++) {
      const z = (i - (count - 1) / 2) * 1.15;
      this.slotScreens.push(kit.slotMachine(slotGroup, 0, z, -Math.PI / 2));
      this.slotLocal.push({ x: slotGroup.position.x, z: slotGroup.position.z + z });
    }
    const disc = kit.luckyWheel(at(S.wheel.x, S.wheel.z - 2.2), luckyWheelTexture());
    kit.bar(at(12, 5.9));
    kit.dealer(this.root, S.cashier.x - 3.0, S.cashier.z, Math.PI / 2);
    return { wheel, ball, disc };
  }

  private buildLights(): void {
    const c = CasinoConfig.room.center;
    const y = CasinoConfig.room.height - 0.6;
    const spots: { x: number; z: number; color: number }[] = [
      { x: c.x - 12, z: c.z + 4, color: 0xffd9a0 },
      { x: c.x + 12, z: c.z + 4, color: 0xffd9a0 },
      { x: c.x - 10, z: c.z - 8, color: 0xffc2d6 },
      { x: c.x + 10, z: c.z - 8, color: 0xbfe9ff },
      { x: c.x, z: c.z - 11, color: 0xffd23f },
    ];
    for (const s of spots) {
      const light = new PointLight(s.color, 90, 34, 2);
      light.position.set(s.x, y, s.z);
      this.root.add(light);
      this.lights.push(light);
    }
  }

  // --- CasinoStage -------------------------------------------------------------

  spinRoulette(pocket: number, seconds: number): void {
    const anim = this.roulette;
    anim.from = this.rouletteWheel.rotation.y;
    anim.to = spinTarget(anim.from, pocketAngle(pocket), CasinoConfig.roulette.wheelTurns);
    anim.duration = Math.max(0.5, seconds);
    anim.t = 0;
    anim.active = true;
  }

  spinWheel(segment: number, seconds: number, onTick?: () => void): void {
    const anim = this.wheel;
    anim.from = this.luckyDisc.rotation.y;
    // The disc stands upright via a +90° X tilt, which puts the texture's angle-0 direction at
    // the bottom; a half turn on top of the segment's texture angle brings it under the pointer.
    anim.to = spinTarget(anim.from, segmentAngle(segment) - Math.PI / WHEEL_SEGMENT_COUNT + Math.PI, CasinoConfig.wheel.turns);
    anim.duration = Math.max(0.5, seconds);
    anim.t = 0;
    anim.lastTick = Math.floor((anim.from / (Math.PI * 2)) * WHEEL_SEGMENT_COUNT);
    anim.onTick = onTick ?? null;
    anim.active = true;
  }

  setSlotScreen(machine: number, reels: readonly SlotSymbol[] | null, spinning: boolean): void {
    const screen = this.slotScreens[machine];
    if (screen) screen.draw(reels, spinning);
  }

  nearestSlot(): number {
    const p = this.host.game.player.position;
    const lx = p.x - this.ctx.origin.x;
    const lz = p.z - this.ctx.origin.z;
    let best = 0;
    let bestD = Infinity;
    for (let i = 0; i < this.slotLocal.length; i++) {
      const dx = this.slotLocal[i].x - lx;
      const dz = this.slotLocal[i].z - lz;
      const d = dx * dx + dz * dz;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  }

  // --- InteriorHandle ------------------------------------------------------------

  update(dt: number): void {
    this.animate(this.roulette, dt, true);
    this.animate(this.wheel, dt, false);
    this.updatePrompt();
  }

  private animate(anim: SpinAnim, dt: number, roulette: boolean): void {
    if (!anim.active) return;
    anim.t = Math.min(anim.duration, anim.t + dt);
    const k = easeOutCubic(anim.t / anim.duration);
    const angle = anim.from + (anim.to - anim.from) * k;
    if (roulette) {
      this.rouletteWheel.rotation.y = angle;
      // The ball counter-rotates and settles on the pointer (world angle 0) as the wheel stops.
      const ballWorld = -(CasinoConfig.roulette.wheelTurns + 2) * Math.PI * 2 * k;
      const local = ballWorld - angle;
      const r = 0.62 + 0.24 * (1 - k);
      this.rouletteBall.position.set(Math.sin(local) * r, 0.1, Math.cos(local) * r);
    } else {
      this.luckyDisc.rotation.y = angle;
      const seg = Math.floor((angle / (Math.PI * 2)) * WHEEL_SEGMENT_COUNT);
      if (seg !== anim.lastTick) {
        anim.lastTick = seg;
        anim.onTick?.();
      }
    }
    if (anim.t >= anim.duration) {
      anim.active = false;
      anim.onTick = null;
    }
  }

  private updatePrompt(): void {
    const game = this.host.game;
    if (!game.systems.has('player') || this.host.isUiOpen()) {
      this.setPrompt(null);
      return;
    }
    const p = game.player.position;
    const lx = p.x - this.ctx.origin.x;
    const lz = p.z - this.ctx.origin.z;
    let best: StationId | null = null;
    let bestD = Infinity;
    for (let i = 0; i < STATION_IDS.length; i++) {
      const id = STATION_IDS[i];
      const s = CasinoConfig.stations[id];
      const dx = s.x - lx;
      const dz = s.z - lz;
      const d = dx * dx + dz * dz;
      if (d < s.radius * s.radius && d < bestD) {
        bestD = d;
        best = id;
      }
    }
    this.setPrompt(best ? STATION_PROMPTS[best] : null);
    if (best && !game.input.isBlocked && game.input.wasPressed('KeyE')) this.host.onStation(best);
  }

  private setPrompt(text: string | null): void {
    const prompts = this.host.game.state.prompts;
    if (text === null) {
      if (this.promptShown) {
        const idx = prompts.indexOf(this.prompt);
        if (idx >= 0) prompts.splice(idx, 1);
        this.promptShown = false;
      }
      return;
    }
    this.prompt.text = text;
    if (!this.promptShown) {
      prompts.push(this.prompt);
      this.promptShown = true;
    }
  }

  onExit(): void {
    this.setPrompt(null);
    this.roulette.active = false;
    this.wheel.active = false;
  }

  dispose(): void {
    this.setPrompt(null);
    for (const s of this.slotScreens) s.dispose();
    for (const l of this.lights) l.dispose();
    this.kit.dispose();
  }
}

/** Stage that forwards to the handle once the framework has built the interior. */
class LazyStage implements CasinoStage {
  handle: CasinoInteriorHandle | null = null;

  get built(): boolean {
    return this.handle !== null;
  }

  spinRoulette(pocket: number, seconds: number): void {
    this.handle?.spinRoulette(pocket, seconds);
  }

  spinWheel(segment: number, seconds: number, onTick?: () => void): void {
    this.handle?.spinWheel(segment, seconds, onTick);
  }

  setSlotScreen(machine: number, reels: readonly SlotSymbol[] | null, spinning: boolean): void {
    this.handle?.setSlotScreen(machine, reels, spinning);
  }

  nearestSlot(): number {
    return this.handle?.nearestSlot() ?? 0;
  }
}

export function createCasinoInterior(host: CasinoInteriorHost): { definition: InteriorDefinition; stage: CasinoStage } {
  const stage = new LazyStage();
  const definition: InteriorDefinition = {
    id: CasinoConfig.interiorId,
    spawnPoint: CasinoConfig.spawnPoint,
    spawnHeading: CasinoConfig.spawnHeading,
    exitTrigger: CasinoConfig.exitTrigger,
    build: (ctx) => {
      const handle = new CasinoInteriorHandle(ctx, host);
      stage.handle = handle;
      return handle;
    },
  };
  return { definition, stage };
}
