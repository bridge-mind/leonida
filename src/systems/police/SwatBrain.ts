import type { WeaponId } from '../../core/peers';
import { PoliceConfig } from './config';
import { CopBrain } from './CopBrain';
import type { PoliceContext } from './context';

/**
 * SWAT / army-tier cop: always lethal (no arrest attempt), tighter accuracy, rifles.
 * At level 6 the "army" response reuses this brain with an extra accuracy bump.
 */
export class SwatBrain extends CopBrain {
  constructor(ctx: PoliceContext, weapon: WeaponId = 'rifle') {
    super(ctx, weapon, true);
    this.provoked = true;
  }

  protected override baseAccuracy(): number {
    return PoliceConfig.cop.accuracySwat + (this.ctx.level >= 6 ? 0.1 : 0);
  }
}
