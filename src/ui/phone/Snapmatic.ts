import type { Game } from '../../core/Game';

/**
 * Snapmatic: grabs the WebGL canvas as a PNG. The renderer does not preserve the
 * drawing buffer, so a frame is rendered synchronously right before `toDataURL`.
 */
export class Snapmatic {
  lastDataUrl: string | null = null;
  count = 0;

  constructor(private readonly game: Game) {}

  capture(): string {
    const renderer = this.game.renderer;
    renderer.render(this.game.scene, this.game.viewCamera);
    const url = renderer.domElement.toDataURL('image/png');
    this.lastDataUrl = url;
    this.count++;
    this.game.audio.play('ui.snapmatic');
    return url;
  }

  /** Triggers a browser download of the last (or a fresh) snapshot. */
  download(): string {
    const url = this.lastDataUrl ?? this.capture();
    const a = document.createElement('a');
    a.href = url;
    a.download = `leonida-snap-${Date.now()}.png`;
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    a.remove();
    return url;
  }
}
