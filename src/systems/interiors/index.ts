import { Config } from '../../core/Config';
import type { Game } from '../../core/Game';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { Interiors, type InteriorDefinition } from '../../world/interiors/Interiors';

/**
 * Exposes the Interiors framework as `game.interiors` and handles the door
 * prompt / E key. Agent 9 owns this folder from Stage 3 and adds content; the
 * framework class itself lives in `src/world/interiors/Interiors.ts`.
 */
export class InteriorsSystem implements GameSystem {
  readonly framework: Interiors;
  private readonly prompt = { key: 'E', text: '' };
  private promptShown = false;

  constructor(private readonly game: Game) {
    this.framework = new Interiors(game);
  }

  init(): void {
    if (this.game.systems.has('world')) this.framework.buildDoorMarkers(this.game.world.landmarks.list);
  }

  // --- Public API (TEAM-PLAN §3.6) -------------------------------------------

  register(def: InteriorDefinition): void {
    this.framework.register(def);
  }

  enter(id: string): Promise<boolean> {
    return this.framework.enter(id);
  }

  exit(): Promise<boolean> {
    return this.framework.exit();
  }

  get current(): string | null {
    return this.framework.current;
  }

  get isInside(): boolean {
    return this.framework.current !== null;
  }

  has(id: string): boolean {
    return this.framework.has(id);
  }

  get registeredIds(): string[] {
    return this.framework.registeredIds;
  }

  // --- Loop ------------------------------------------------------------------

  update(dt: number): void {
    this.framework.update(dt);
    if (!this.game.systems.has('player')) return;
    const player = this.game.player;
    if (player.isMounted || !player.isAlive || this.framework.transitioning) {
      this.setPrompt(null);
      return;
    }
    const pos = player.position;
    if (this.framework.current) {
      const exit = this.framework.currentExitTrigger();
      const near = exit !== null && exit.distanceToSquared(pos) < Config.interiors.doorRadius ** 2 * 1.5;
      this.setPrompt(near ? 'Leave' : null);
      if (near && this.game.input.wasPressed('KeyE')) void this.framework.exit();
      return;
    }
    const door = this.framework.doorNear(pos, Config.interiors.doorRadius);
    if (!door || !door.interiorId) {
      this.setPrompt(null);
      return;
    }
    const registered = this.framework.has(door.interiorId);
    this.setPrompt(registered ? `Enter ${door.name}` : `${door.name} (closed)`);
    if (this.game.input.wasPressed('KeyE')) {
      if (registered) void this.framework.enter(door.interiorId);
      else this.game.events.emit('notification', { text: `${door.name} is closed.`, kind: 'info' });
    }
  }

  private setPrompt(text: string | null): void {
    const prompts = this.game.state.prompts;
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

  dispose(): void {
    this.setPrompt(null);
    this.framework.dispose();
  }

  debugActions = {
    enter: (id: unknown): Promise<boolean> => this.enter(String(id)),
    exit: (): Promise<boolean> => this.exit(),
    list: (): string[] => this.registeredIds,
    current: (): string | null => this.current,
  };
}

const module: SystemModule = {
  id: 'interiors',
  priority: PRIORITY.interiors,
  create: (game) => new InteriorsSystem(game),
};

export default module;
