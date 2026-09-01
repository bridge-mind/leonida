import type { Events } from './Events';
import type { Time } from './Time';

export interface Serializable {
  serialize?(): unknown;
  deserialize?(data: unknown): void;
}

interface SaveFile {
  version: number;
  savedAt: number;
  time: { hour: number; day: number };
  systems: Record<string, unknown>;
}

/**
 * Registry over systems' `serialize()/deserialize()` hooks. One versioned JSON
 * blob in localStorage per slot. Systems are registered automatically by Game.
 */
export class Save {
  static readonly VERSION = 1;
  static readonly KEY_PREFIX = 'leonida.save.';

  private readonly registry = new Map<string, Serializable>();

  constructor(
    private readonly events: Events,
    private readonly time: Time,
  ) {}

  register(id: string, target: Serializable): void {
    if (target.serialize || target.deserialize) this.registry.set(id, target);
  }

  unregister(id: string): void {
    this.registry.delete(id);
  }

  private key(slot: string): string {
    return `${Save.KEY_PREFIX}${slot}`;
  }

  hasSave(slot = 'main'): boolean {
    try {
      return localStorage.getItem(this.key(slot)) !== null;
    } catch {
      return false;
    }
  }

  save(slot = 'main'): boolean {
    const file: SaveFile = { version: Save.VERSION, savedAt: Date.now(), time: this.time.serialize(), systems: {} };
    for (const [id, target] of this.registry) {
      if (!target.serialize) continue;
      try {
        file.systems[id] = target.serialize();
      } catch (err) {
        console.error(`[save:${id}] serialize failed`, err);
      }
    }
    try {
      localStorage.setItem(this.key(slot), JSON.stringify(file));
    } catch (err) {
      console.error('[save] localStorage write failed', err);
      return false;
    }
    this.events.emit('save:saved', { slot });
    return true;
  }

  load(slot = 'main'): boolean {
    let raw: string | null = null;
    try {
      raw = localStorage.getItem(this.key(slot));
    } catch {
      return false;
    }
    if (!raw) return false;
    let file: SaveFile;
    try {
      file = JSON.parse(raw) as SaveFile;
    } catch (err) {
      console.error('[save] corrupt save file', err);
      return false;
    }
    if (file.version !== Save.VERSION) {
      console.warn(`[save] version ${file.version} != ${Save.VERSION}; ignoring save`);
      return false;
    }
    this.time.deserialize(file.time);
    for (const [id, target] of this.registry) {
      if (!target.deserialize || !(id in file.systems)) continue;
      try {
        target.deserialize(file.systems[id]);
      } catch (err) {
        console.error(`[save:${id}] deserialize failed`, err);
      }
    }
    this.events.emit('save:loaded', { slot });
    return true;
  }

  clear(slot = 'main'): void {
    try {
      localStorage.removeItem(this.key(slot));
    } catch {
      /* storage unavailable */
    }
  }

  /** Snapshot without writing to storage (used by tests and autosave diffing). */
  snapshot(): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    for (const [id, target] of this.registry) if (target.serialize) out[id] = target.serialize();
    return out;
  }
}
