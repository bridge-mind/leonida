import type { Game } from '../../core/Game';
import type { CardKind, DialogueLine, IUiPeer } from '../../core/peers';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { Cards } from '../../ui/Cards';
import { Dialogue } from '../../ui/Dialogue';
import { el, toggleClass, type UiComponent } from '../../ui/dom';
import { FullMap } from '../../ui/FullMap';
import { HUD } from '../../ui/HUD';
import { MapLayer } from '../../ui/MapLayer';
import { Minimap } from '../../ui/Minimap';
import { Notifications, type NotificationKind } from '../../ui/Notifications';
import { Objective } from '../../ui/Objective';
import { PauseMenu } from '../../ui/PauseMenu';
import { ProgressBar } from '../../ui/ProgressBar';
import { Prompts } from '../../ui/Prompts';
import { TitleScreen } from '../../ui/TitleScreen';
import { UiConfig } from './config';

/** URL params that mean "a test or isolation run needs the game live immediately". */
const NO_TITLE_PARAMS = ['notitle', 'systems', 'autotest', 'actions'];
/** Set before a reload so the next boot skips the title and starts a fresh game. */
const FRESH_START_KEY = 'leonida.freshStart';

/**
 * Mounts every overlay under `#ui`, feeds them `game.state` each frame and owns the
 * modal stack (title → pause → full map). Implements `IUiPeer` for other systems.
 */
export class UiSystem implements GameSystem, IUiPeer {
  private root!: HTMLElement;
  private readonly components: UiComponent[] = [];
  private hud!: HUD;
  private minimap: Minimap | null = null;
  private fullMap: FullMap | null = null;
  private notifications!: Notifications;
  private pauseMenu!: PauseMenu;
  private title!: TitleScreen;
  /** True once gameplay has begun this page load (New Game after that reloads). */
  private sessionStarted = false;
  private cards!: Cards;
  private dialogueBox!: Dialogue;
  private progress!: ProgressBar;
  private readonly uiPrompts = new Map<string, { key: string; text: string }>();
  private paused = false;
  private savedScale = 1;
  private savedClockRunning = true;
  /** True while we asked the browser to release the pointer, so the change event is not read as "user pressed Esc". */
  private expectUnlock = false;
  /**
   * Seconds during which Esc is ignored right after pausing: some browsers both drop the
   * pointer lock (our pause trigger) and deliver the Esc keydown, which would resume at once.
   */
  private escGrace = 0;
  private readonly offs: (() => void)[] = [];
  readonly dialogue: IUiPeer['dialogue'];

  constructor(private readonly game: Game) {
    const self = this;
    this.dialogue = {
      show: (lines: DialogueLine[]): Promise<void> => self.dialogueBox.show(lines),
      hide: (): void => self.dialogueBox.hide(),
      get active(): boolean {
        return self.dialogueBox?.active ?? false;
      },
    };
  }

  init(): void {
    const game = this.game;
    this.root = document.getElementById('ui') ?? document.body.appendChild(el('div'));
    this.root.id = 'ui';
    this.root.classList.add('ui-root');

    this.notifications = new Notifications(game);
    this.hud = new HUD(game);
    const bottom = el('div', 'bottom-stack');
    const objective = new Objective();
    const prompts = new Prompts();
    this.progress = new ProgressBar();
    bottom.append(objective.el, prompts.el, this.progress.el);
    this.dialogueBox = new Dialogue(game);
    this.cards = new Cards(game);
    this.pauseMenu = new PauseMenu(game, {
      resume: () => this.resume(),
      openMap: () => this.openMap(),
      quitToTitle: () => this.quitToTitle(),
      notify: (text, kind) => this.notify(text, kind),
      relockPointer: () => this.lockPointer(),
    });
    this.title = new TitleScreen(game, {
      newGame: () => {
        // Systems have no reset; after a session has started, a fresh world needs a reload.
        if (this.sessionStarted) {
          game.save.clear();
          sessionStorage.setItem(FRESH_START_KEY, '1');
          location.reload();
          return;
        }
        this.leaveTitle('ui:newGame');
      },
      continueGame: () => {
        game.save.load();
        this.leaveTitle('ui:continue');
      },
    });
    this.components.push(this.notifications, this.hud, objective, prompts, this.progress, this.dialogueBox, this.cards, this.pauseMenu, this.title);
    this.root.append(this.notifications.el, this.hud.el, bottom, this.dialogueBox.el, this.cards.el);

    if (game.systems.has('world')) {
      const layer = new MapLayer(game.world.roadGraph, game.world.landmarks.list);
      this.minimap = new Minimap(game, layer);
      this.fullMap = new FullMap(game, layer);
      this.fullMap.onClose = () => {
        game.input.popBlocker('map');
        if (!this.paused) this.lockPointer();
      };
      this.components.push(this.minimap, this.fullMap);
      this.root.append(this.minimap.el, this.fullMap.el);
    }
    this.root.append(this.pauseMenu.el, this.title.el);

    this.offs.push(
      game.events.on('player:died', () => {
        // `Arrest.bust()` emits player:busted synchronously after `kill('busted')` → player:died;
        // defer so the Busted card (and its sting) claims the slot and Wasted never flashes.
        queueMicrotask(() => {
          if (this.cards.activeKind !== 'busted') void this.card('wasted', 'Wasted');
        });
      }),
      game.events.on('player:busted', () => void this.card('busted', 'Busted')),
      // Pausing releases the pointer, which releases Tab and closes the wheel while scale is 0;
      // the wheel cannot undo its slow-mo then, so drop it from the scale we restore on resume.
      game.events.on('weapon:wheelToggled', ({ open }) => {
        if (!open && this.paused && this.savedScale < 1) this.savedScale = 1;
      }),
    );
    document.addEventListener('pointerlockchange', this.onPointerLockChange);

    if (sessionStorage.getItem(FRESH_START_KEY)) {
      sessionStorage.removeItem(FRESH_START_KEY);
      this.sessionStarted = true;
      game.events.emit('ui:newGame', {});
      this.lockPointer();
    } else if (!NO_TITLE_PARAMS.some((p) => game.params.has(p))) {
      this.showTitle();
    } else {
      this.sessionStarted = true;
    }
  }

  // --- IUiPeer ---------------------------------------------------------------------

  notify(text: string, kind: NotificationKind = 'info'): void {
    this.notifications.post(text, kind);
  }

  card(kind: CardKind, text: string, seconds: number = UiConfig.cards.defaultSeconds): Promise<void> {
    return this.cards.show(kind, text, seconds);
  }

  prompt(key: string, text: string): void {
    const existing = this.uiPrompts.get(key);
    if (existing) {
      existing.text = text;
      return;
    }
    const p = { key, text };
    this.uiPrompts.set(key, p);
    this.game.state.prompts.push(p);
  }

  clearPrompt(key: string): void {
    const p = this.uiPrompts.get(key);
    if (!p) return;
    this.uiPrompts.delete(key);
    const list = this.game.state.prompts;
    const idx = list.indexOf(p);
    if (idx >= 0) list.splice(idx, 1);
  }

  setObjective(text: string | null): void {
    const changed = this.game.state.objective !== text;
    this.game.state.objective = text;
    if (text && changed) this.game.audio.play('mission.objective');
  }

  readonly progressBar = {
    show: (label: string, fraction: number): void => this.progress.show(label, fraction),
    hide: (): void => this.progress.hide(),
  };

  letterbox(on: boolean): void {
    this.cards.letterbox(on);
  }

  pause(): void {
    if (this.paused) return;
    const time = this.game.time;
    this.paused = true;
    this.savedScale = time.scale;
    this.savedClockRunning = time.clockRunning;
    time.scale = 0;
    time.clockRunning = false;
    this.escGrace = 0.3;
    this.game.input.pushBlocker('pause');
    this.releasePointer();
    this.pauseMenu.show();
    this.game.audio.play('ui.pause');
    this.game.events.emit('ui:paused', {});
  }

  resume(): void {
    if (!this.paused) return;
    const time = this.game.time;
    this.paused = false;
    time.scale = this.savedScale;
    time.clockRunning = this.savedClockRunning;
    this.pauseMenu.hide();
    this.game.input.popBlocker('pause');
    this.game.audio.play('ui.unpause');
    this.game.events.emit('ui:resumed', {});
    this.lockPointer();
  }

  get isPaused(): boolean {
    return this.paused;
  }

  get isModal(): boolean {
    return this.title.visible || this.pauseMenu.visible || (this.fullMap?.visible ?? false);
  }

  // --- Modal flow ----------------------------------------------------------------

  private showTitle(): void {
    if (this.title.visible) return;
    this.game.input.pushBlocker('title');
    this.releasePointer();
    this.title.show();
  }

  private leaveTitle(event: 'ui:newGame' | 'ui:continue'): void {
    const game = this.game;
    this.sessionStarted = true;
    this.title.hide();
    if (game.systems.has('camera')) {
      const cam = game.camera;
      cam.cinematic.stop();
      const mounted = game.systems.has('player') && game.player.isMounted;
      cam.setMode(mounted ? 'vehicle' : 'thirdPerson');
      cam.snap();
    }
    game.input.popBlocker('title');
    game.audio.play('ui.title.stinger');
    game.events.emit(event, {});
    this.lockPointer();
  }

  /** Only called from click/Enter handlers: the browser needs a user gesture to grant pointer lock. */
  private lockPointer(): void {
    if (this.game.input.pointerLocked || this.isModal) return;
    try {
      void Promise.resolve(this.game.canvas.requestPointerLock()).catch(() => {});
    } catch {
      /* not allowed without a gesture; the canvas mousedown handler retries later */
    }
  }

  private quitToTitle(): void {
    this.fullMap?.close();
    // Title first so `resume()` sees a modal and does not re-lock the pointer.
    this.showTitle();
    this.resume();
  }

  private openMap(): void {
    if (!this.fullMap || this.fullMap.visible) return;
    this.game.input.pushBlocker('map');
    this.releasePointer();
    this.fullMap.open();
  }

  private releasePointer(): void {
    if (!this.game.input.pointerLocked) return;
    this.expectUnlock = true;
    this.game.input.exitPointerLock();
  }

  /** Browsers swallow Esc while the pointer is locked; losing the lock is our pause signal. */
  private readonly onPointerLockChange = (): void => {
    if (document.pointerLockElement) return;
    if (this.expectUnlock || this.game.input.unlockRequested) {
      this.expectUnlock = false;
      this.game.input.unlockRequested = false;
      return;
    }
    if (this.game.ready && !this.isModal) this.pause();
  };

  private handleInput(): void {
    const input = this.game.input;
    if (this.title.visible) {
      this.title.handleInput();
      return;
    }
    if (this.fullMap?.visible) {
      if (input.wasPressed('Escape') || input.wasPressed('KeyM')) this.fullMap.close();
      return;
    }
    if (this.pauseMenu.visible) {
      if (input.wasPressed('Escape') && this.escGrace <= 0) this.resume();
      else this.pauseMenu.handleInput();
      return;
    }
    if (this.dialogueBox.active && input.wasPressed('Space')) this.dialogueBox.advance();
    // Another modal (weapon wheel, casino panel, phone, shop) owns input: Esc/M belong to it.
    if (input.isBlocked) return;
    if (input.wasPressed('Escape')) {
      this.pause();
      return;
    }
    if (input.wasPressed('KeyM')) this.openMap();
    if (input.wasPressed('KeyP')) this.game.save.save();
  }

  // --- Loop ----------------------------------------------------------------------

  update(): void {
    const realDt = this.game.time.realDt;
    if (this.escGrace > 0) this.escGrace -= realDt;
    this.handleInput();
    toggleClass(this.root, 'menu-open', this.isModal);
    const state = this.game.state;
    // Menus animate on real time; HUD-family overlays freeze with the simulation while paused.
    const hudDt = this.paused ? 0 : realDt;
    for (const c of this.components) {
      c.update(state, c === this.pauseMenu || c === this.title || c === this.fullMap ? realDt : hudDt);
    }
  }

  dispose(): void {
    for (const off of this.offs) off();
    document.removeEventListener('pointerlockchange', this.onPointerLockChange);
    this.resume();
    for (const key of Array.from(this.uiPrompts.keys())) this.clearPrompt(key);
    for (const c of this.components) c.dispose?.();
    this.root.classList.remove('ui-root', 'menu-open');
  }

  debugActions = {
    notify: (text: unknown, kind?: unknown): void => this.notify(String(text ?? 'Notification'), asKind(kind)),
    card: (kind: unknown, text?: unknown, seconds?: unknown): Promise<void> =>
      this.card(asCardKind(kind), String(text ?? ''), seconds === undefined ? undefined : Number(seconds)),
    dialogue: (): Promise<void> =>
      this.dialogue.show([
        { speaker: 'Lucia', text: 'You sure about this, Jason?' },
        { speaker: 'Jason', text: "We're broke. Of course I'm sure." },
        { speaker: 'Lucia', text: 'Fine. But we do it my way.' },
      ]),
    objective: (text: unknown): void => this.setObjective(text === undefined || text === null || text === '' ? null : String(text)),
    progress: (fraction: unknown): void => {
      const f = Number(fraction);
      if (!Number.isFinite(f) || f < 0) this.progressBar.hide();
      else this.progressBar.show('Filling bag', f);
    },
    title: (): void => this.showTitle(),
    pause: (): void => this.pause(),
    resume: (): void => this.resume(),
    map: (): boolean => {
      if (!this.fullMap) return false;
      if (this.fullMap.visible) this.fullMap.close();
      else this.openMap();
      return this.fullMap.visible;
    },
    state: (): Record<string, unknown> => ({
      isPaused: this.paused,
      isModal: this.isModal,
      title: this.title.visible,
      pause: this.pauseMenu.visible,
      fullMap: this.fullMap?.visible ?? false,
      dialogue: this.dialogueBox.active,
      card: this.cards.isActive,
      notifications: this.notifications.count,
      blockers: this.game.input.isBlocked,
      timeScale: this.game.time.scale,
    }),
  };
}

function asKind(v: unknown): NotificationKind {
  return v === 'money' || v === 'warning' ? v : 'info';
}

function asCardKind(v: unknown): CardKind {
  switch (v) {
    case 'wasted':
    case 'busted':
    case 'missionPassed':
    case 'missionFailed':
      return v;
    default:
      return 'custom';
  }
}

const module: SystemModule = {
  id: 'ui',
  priority: PRIORITY.ui,
  create: (game) => new UiSystem(game),
};

export default module;
