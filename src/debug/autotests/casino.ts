import type { Game } from '../../core/Game';
import type { CasinoSystem } from '../../systems/casino';
import { CasinoConfig } from '../../systems/casino/config';
import type { BlackjackPanel } from '../../ui/casino/BlackjackPanel';
import type { RoulettePanel } from '../../ui/casino/RoulettePanel';
import type { SlotsPanel } from '../../ui/casino/SlotsPanel';
import type { WheelPanel } from '../../ui/casino/WheelPanel';
import { assert, Suite, waitFor, waitFrames } from './util';

/**
 * Casino acceptance (docs/TEAM-PLAN.md Agent 6).
 * Run: npm run smoke -- --systems=world,player,camera,economy,interiors,casino --name=casino --autotest=casino
 */
export default async function casino(game: Game): Promise<void> {
  const suite = new Suite('casino', game);
  const system = game.systems.get('casino') as CasinoSystem | undefined;
  assert(system, 'casino system missing');
  const economy = game.economy;
  assert(economy, 'economy system missing');
  const chips = (): number => game.debug.run('casino.chips') as number;
  const panel = <T>(id: string): T => {
    const p = system.panel;
    assert(p && p.id.startsWith(id), `expected the ${id} panel to be open, got ${p?.id ?? 'none'}`);
    return p as unknown as T;
  };
  const closePanel = async (): Promise<void> => {
    game.debug.run('casino.close');
    await waitFrames(game, 2);
    assert(system.panel === null, 'panel did not close');
    assert(!game.input.isBlocked, 'input blocker left behind');
  };

  await suite.check('enter-interior', async () => {
    game.debug.run('casino.unlock');
    await game.debug.run('casino.enter');
    await waitFor(() => game.state.interior === 'casino', 6000, 'interior:entered casino');
    assert(system.isInside, 'system did not register casino:entered');
    assert(system.stage.built, 'casino interior was not built');
  });

  await suite.check('buy-chips', async () => {
    if (economy.cash < 1000) economy.add(1000 - economy.cash + 500, 'test');
    const before = economy.cash;
    const start = chips();
    game.debug.run('casino.buy', 1000);
    await waitFrames(game, 1);
    assert(economy.cash === before - 1000, `cash ${economy.cash}, expected ${before - 1000}`);
    assert(chips() === start + 1000, `chips ${chips()}, expected ${start + 1000}`);
    assert(game.state.chips === chips(), 'game.state.chips not published');
    if (start > 0) game.debug.run('casino.cashout', start);
    assert(chips() === 1000, `chips should be 1000, got ${chips()}`);
  });

  await suite.check('blackjack-natural-pays-3-to-2', async () => {
    game.debug.run('casino.rig', 'blackjack', ['AS', '9D', 'KH', '7C']);
    game.debug.run('casino.play', 'blackjack');
    await waitFrames(game, 2);
    const bj = panel<BlackjackPanel>('blackjack');
    bj.setBet(100);
    const bets = document.querySelector<HTMLButtonElement>('#ui [data-panel^="blackjack"] [data-action="deal"]');
    assert(bets, 'Deal button missing');
    bets.click();
    assert(chips() === 900, `bet not deducted (chips ${chips()})`);
    await waitFor(() => chips() === 1150, 5000, 'blackjack payout of 250 (bet 100 back + 150)');
    assert(document.querySelectorAll('#ui .bj-card').length === 4, 'expected four cards on the table');
    const msg = document.querySelector('#ui [data-panel^="blackjack"] .casino-msg')?.textContent ?? '';
    assert(msg.includes('BLACKJACK'), `outcome text '${msg}'`);
    await closePanel();
  });

  await suite.check('roulette-straight-17', async () => {
    game.debug.run('casino.rig', 'roulette', 17);
    game.debug.run('casino.play', 'roulette');
    await waitFrames(game, 2);
    const rl = panel<RoulettePanel>('roulette');
    const before = chips();
    rl.selectChip(5);
    const cell = document.querySelector<HTMLElement>('#ui [data-panel="roulette"] [data-cell="p17"]');
    assert(cell, 'number 17 cell missing');
    cell.click();
    cell.click();
    assert(rl.bets.length === 1 && rl.bets[0].amount === 10, 'two 5-chip clicks should merge into a 10 straight bet');
    assert(chips() === before - 10, 'bet not deducted');
    rl.spin();
    await waitFor(() => chips() === before + 350, (CasinoConfig.roulette.spinSeconds + 3) * 1000, 'roulette payout of 360');
    const shown = document.querySelector('#ui [data-panel="roulette"] [data-field="outcome"]')?.textContent;
    assert(shown === '17', `display shows '${shown}'`);
    await closePanel();
  });

  await suite.check('slots-jackpot', async () => {
    game.debug.run('casino.rig', 'slots', ['diamond', 'diamond', 'diamond']);
    game.debug.run('casino.play', 'slots');
    await waitFrames(game, 2);
    const sl = panel<SlotsPanel>('slots');
    const before = chips();
    sl.setBet(10);
    sl.spin();
    assert(chips() === before - 10, 'bet not deducted');
    await waitFor(() => chips() === before - 10 + 5000, 6000, 'slots jackpot of 5000');
    const reels = Array.from(document.querySelectorAll<HTMLElement>('#ui .sl-reel')).map((r) => r.dataset.symbol);
    assert(reels.join(',') === 'diamond,diamond,diamond', `reels show ${reels.join(',')}`);
    assert(sl.lastResult?.jackpot === true, 'result not flagged as jackpot');
    await closePanel();
  });

  await suite.check('wheel-once-per-day', async () => {
    system.wheel.lastSpinDay = 0;
    game.debug.run('casino.rig', 'wheel', 5); // $2,500 cash segment
    game.debug.run('casino.play', 'wheel');
    await waitFrames(game, 2);
    const wheel = panel<WheelPanel>('wheel');
    const cash = economy.cash;
    assert(wheel.spin(), 'first spin refused');
    await waitFor(() => economy.cash === cash + 2500, (CasinoConfig.wheel.spinSeconds + 3) * 1000, 'wheel cash prize');
    assert(wheel.spin() === false, 'second spin on the same day was accepted');
    assert(!system.wheel.canSpin(game.time.day), 'canSpin still true after spinning today');
    await closePanel();
  });

  await suite.check('events-and-state', async () => {
    let lost = 0;
    const off = game.events.on('casino:lost', () => lost++);
    game.debug.run('casino.rig', 'roulette', 0);
    game.debug.run('casino.play', 'roulette');
    await waitFrames(game, 2);
    const rl = panel<RoulettePanel>('roulette');
    rl.selectChip(5);
    rl.placeOutside('red');
    rl.spin();
    await waitFor(() => lost === 1, (CasinoConfig.roulette.spinSeconds + 3) * 1000, 'casino:lost event');
    off();
    await closePanel();
    assert(game.state.chips === chips(), 'state.chips out of sync');
  });

  await suite.check('cashout-and-exit', async () => {
    const cash = economy.cash;
    const bal = chips();
    game.debug.run('casino.cashout', bal);
    assert(chips() === 0 && economy.cash === cash + bal, 'cash out did not move chips to cash');
    const interiors = game.interiors;
    assert(interiors, 'interiors peer missing');
    await interiors.exit();
    await waitFor(() => game.state.interior === null, 6000, 'interior exit');
    assert(!system.isInside, 'system still thinks the player is inside');
  });

  await suite.check('no-errors-after-run', () => {
    assert(game.debug.errorCount === 0, `${game.debug.errorCount} console errors: ${game.debug.errors.join(' | ')}`);
  });

  suite.finish();
}
