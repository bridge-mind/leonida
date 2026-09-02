import type { V3 } from './entities';

/**
 * Keyboard/mouse state with pointer lock, plus a scripted layer used by autotests.
 * Scripted presses and real keys are merged: `isDown(code)` is true if either holds it.
 *
 * Mouse buttons map to codes 'Mouse0' (LMB), 'Mouse1' (MMB), 'Mouse2' (RMB).
 */
export class Input {
  private readonly down = new Set<string>();
  private readonly pressedThisFrame = new Set<string>();
  private readonly releasedThisFrame = new Set<string>();
  private readonly scripted = new Map<string, number>(); // code -> release time (ms, Infinity = hold)
  private readonly scriptedPrev = new Set<string>();

  /** Accumulated mouse movement since the last `endFrame()`. */
  mouseDeltaX = 0;
  mouseDeltaY = 0;
  /** Accumulated wheel delta since last frame (positive = scroll down). */
  wheelDelta = 0;
  pointerLocked = false;
  /** Set by `exitPointerLock()` so the UI can tell a deliberate release from the user pressing Esc. */
  unlockRequested = false;
  /** When false, real keyboard/mouse input is ignored (UI menus use this). Scripted input still works. */
  enabled = true;

  private scriptedAim = false;
  private scriptedFire = false;
  private scriptedMouseWorld: V3 | null = null;
  private readonly blockers = new Set<string>();

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.bind();
  }

  private bind(): void {
    const c = this.canvas;
    window.addEventListener('keydown', (e) => {
      if (!this.enabled) return;
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return;
      if (e.code === 'Tab' || e.code === 'F3' || (e.code.startsWith('Digit') && e.altKey)) e.preventDefault();
      if (!this.down.has(e.code)) this.pressedThisFrame.add(e.code);
      this.down.add(e.code);
    });
    window.addEventListener('keyup', (e) => {
      if (this.down.has(e.code)) this.releasedThisFrame.add(e.code);
      this.down.delete(e.code);
    });
    window.addEventListener('blur', () => this.releaseAll());
    c.addEventListener('mousedown', (e) => {
      if (!this.pointerLocked && this.enabled) this.requestPointerLock();
      if (!this.enabled) return;
      const code = `Mouse${e.button}`;
      if (!this.down.has(code)) this.pressedThisFrame.add(code);
      this.down.add(code);
    });
    window.addEventListener('mouseup', (e) => {
      const code = `Mouse${e.button}`;
      if (this.down.has(code)) this.releasedThisFrame.add(code);
      this.down.delete(code);
    });
    window.addEventListener('mousemove', (e) => {
      if (!this.pointerLocked || !this.enabled) return;
      this.mouseDeltaX += e.movementX;
      this.mouseDeltaY += e.movementY;
    });
    c.addEventListener('wheel', (e) => {
      if (!this.enabled) return;
      this.wheelDelta += Math.sign(e.deltaY);
      e.preventDefault();
    }, { passive: false });
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    document.addEventListener('pointerlockchange', () => {
      this.pointerLocked = document.pointerLockElement === c;
      if (this.pointerLocked) this.unlockRequested = false;
      else this.releaseAll();
    });
  }

  private releaseAll(): void {
    for (const code of this.down) this.releasedThisFrame.add(code);
    this.down.clear();
  }

  // --- Queries -------------------------------------------------------------

  isDown(code: string): boolean {
    if (this.down.has(code)) return true;
    const until = this.scripted.get(code);
    return until !== undefined && until > performance.now();
  }

  /** True only on the frame the key went down. */
  wasPressed(code: string): boolean {
    return this.pressedThisFrame.has(code);
  }

  wasReleased(code: string): boolean {
    return this.releasedThisFrame.has(code);
  }

  /** -1..1 on each axis from WASD: x = right, y = forward. Arrows are reserved (Up = phone, menus). */
  moveAxis(out: { x: number; y: number }): { x: number; y: number } {
    let x = 0;
    let y = 0;
    if (this.isDown('KeyW')) y += 1;
    if (this.isDown('KeyS')) y -= 1;
    if (this.isDown('KeyD')) x += 1;
    if (this.isDown('KeyA')) x -= 1;
    out.x = x;
    out.y = y;
    return out;
  }

  get aim(): boolean {
    return this.scriptedAim || this.isDown('Mouse2');
  }

  get fire(): boolean {
    return this.scriptedFire || this.isDown('Mouse0');
  }

  /** World-space aim point injected by autotests; null when real mouse aim applies. */
  get mouseWorld(): V3 | null {
    return this.scriptedMouseWorld;
  }

  /** UI components register a blocker while open; systems should check `isBlocked`. */
  pushBlocker(id: string): void {
    this.blockers.add(id);
  }

  popBlocker(id: string): void {
    this.blockers.delete(id);
  }

  get isBlocked(): boolean {
    return this.blockers.size > 0;
  }

  requestPointerLock(): void {
    if (this.pointerLocked) return;
    // Browsers reject when the gesture was consumed elsewhere (menus, fresh focus); not an error.
    const result = (this.canvas.requestPointerLock as () => Promise<void> | void).call(this.canvas);
    if (result && typeof (result as Promise<void>).catch === 'function') {
      (result as Promise<void>).catch(() => undefined);
    }
  }

  exitPointerLock(): void {
    if (!this.pointerLocked) return;
    this.unlockRequested = true;
    document.exitPointerLock();
  }

  /** Drops this frame's press edge for `code` (a modal that handled the key via raw DOM events). */
  consume(code: string): void {
    this.pressedThisFrame.delete(code);
  }

  // --- Scripted layer (autotests) -----------------------------------------

  press(code: string, ms = 80): void {
    this.scripted.set(code, performance.now() + ms);
  }

  hold(code: string): void {
    this.scripted.set(code, Infinity);
  }

  release(code: string): void {
    this.scripted.delete(code);
    this.releasedThisFrame.add(code);
  }

  look(dx: number, dy: number): void {
    this.mouseDeltaX += dx;
    this.mouseDeltaY += dy;
  }

  setAim(on: boolean): void {
    this.scriptedAim = on;
  }

  setFire(on: boolean): void {
    this.scriptedFire = on;
  }

  setMouseWorld(point: V3 | null): void {
    this.scriptedMouseWorld = point ? { x: point.x, y: point.y, z: point.z } : null;
  }

  // --- Frame lifecycle (Game calls these) ----------------------------------

  /** Promote scripted key transitions into pressed/released edges. */
  beginFrame(): void {
    const now = performance.now();
    for (const [code, until] of this.scripted) {
      if (until <= now) {
        this.scripted.delete(code);
        if (this.scriptedPrev.has(code)) this.releasedThisFrame.add(code);
        this.scriptedPrev.delete(code);
      } else if (!this.scriptedPrev.has(code)) {
        this.pressedThisFrame.add(code);
        this.scriptedPrev.add(code);
      }
    }
    for (const code of this.scriptedPrev) {
      if (!this.scripted.has(code)) {
        this.releasedThisFrame.add(code);
        this.scriptedPrev.delete(code);
      }
    }
  }

  endFrame(): void {
    this.pressedThisFrame.clear();
    this.releasedThisFrame.clear();
    this.mouseDeltaX = 0;
    this.mouseDeltaY = 0;
    this.wheelDelta = 0;
  }
}
