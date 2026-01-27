import asyncio
import time
from datetime import datetime
from trading_bot.data_manager import DataManager
from trading_bot.screener import Screener
from trading_bot.analyzer import MultiKernelAnalyzer
from trading_bot.executor import Executor
from trading_bot.config import config
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TradingBotDryRun:
    """DRY RUN версия бота - без реальных сделок"""

    def __init__(self):
        self.dm = DataManager()
        self.screener = Screener(self.dm)
        self.analyzer = MultiKernelAnalyzer(self.dm)
        self.executor = Executor(self.dm)
        self.running = False

    async def run_cycle(self):
        """Один цикл сканирования (DRY RUN)"""
        cycle_start = time.time()

        logger.info("=" * 60)
        logger.info(f"🔄 DRY RUN SCAN CYCLE START | {datetime.now()}")
        logger.info("=" * 60)

        # 1. Получаем топ монет
        symbols = self.screener.get_symbols_to_analyze()

        if not symbols:
            logger.info("No symbols match criteria")
            return

        logger.info(f"Analyzing {len(symbols)} symbols...")

        # 2. Анализируем каждый символ
        signals = []
        for symbol in symbols:
            try:
                signal = self.analyzer.analyze(symbol)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")

            await asyncio.sleep(0.1)  # Rate limiting

        # 3. Сортируем по силе сигнала
        signals.sort(key=lambda x: x.strength, reverse=True)

        if signals:
            logger.info(f"\n📊 SIGNALS FOUND: {len(signals)}")
            for s in signals:
                zone = "🟢 GREEN" if s.is_outperforming else "🔴 RED"
                logger.info(f"   {s.action} {s.symbol} | {zone} | strength={s.strength:.2f}")
        else:
            logger.info("\n📊 No signals found")

        # 4. DRY RUN: Показываем, что БЫ исполнили
        executed = 0
        for signal in signals[:2]:  # Максимум 2 сделки за цикл
            # Вместо реального исполнения - только логируем
            logger.info(f"🔍 DRY RUN: Would execute {signal.action} {signal.symbol}")
            logger.info(f"   Qty: ~{self.executor.calculate_position_size(signal.symbol, signal.price, signal.stop_loss):.4f}")
            logger.info(f"   SL: {self.executor.round_price(signal.symbol, signal.stop_loss)}")
            logger.info(f"   TP: {self.executor.round_price(signal.symbol, signal.take_profit)}")
            logger.info(f"   Reason: {signal.reason}")
            executed += 1

        # 5. Мониторинг позиций
        await self.monitor_positions()

        # Очищаем кеш
        self.analyzer._benchmark_cache.clear()
        self.dm.clear_cache()

        cycle_time = time.time() - cycle_start
        logger.info(f"\n⏱️ DRY RUN Cycle completed in {cycle_time:.2f}s | Would execute: {executed}")

    async def monitor_positions(self):
        """Мониторинг открытых позиций"""
        positions = self.dm.get_positions()

        if positions:
            logger.info(f"\n📈 OPEN POSITIONS: {len(positions)}")
            for pos in positions:
                pnl = pos['unrealized_pnl']
                pnl_pct = (pnl / (pos['size'] * pos['entry_price'])) * 100

                # Получаем статус зоны
                is_green, perf = self.analyzer.get_zone_status(pos['symbol'])
                zone = "🟢" if is_green else "🔴"

                logger.info(
                    f"   {zone} {pos['symbol']} | {pos['side']} | "
                    f"size={pos['size']} | PnL={pnl:.2f} ({pnl_pct:+.2f}%)"
                )
        else:
            logger.info("\n📈 No open positions")

    async def run(self, cycles=1):
        """Запуск DRY RUN на указанное количество циклов"""
        self.running = True

        logger.info("🚀 TRADING BOT DRY RUN STARTED")
        logger.info(f"   Kernel: {config.KERNEL_TYPE} (bandwidth={config.BANDWIDTH})")
        logger.info(f"   Benchmark: {config.BENCHMARK_SYMBOL}")
        logger.info(f"   Timeframe: {config.TIMEFRAME}m")
        logger.info(f"   SL: {config.SL_PERCENT}% | TP: {config.TP_PERCENT}%")
        logger.info(f"   Leverage: {config.LEVERAGE}x")
        logger.info(f"   Cycles to run: {cycles}")
        logger.info("=" * 60)

        for cycle in range(cycles):
            if cycle > 0:
                logger.info(f"\n⏰ Starting cycle {cycle + 1}/{cycles}")

            try:
                await self.run_cycle()

                if cycle < cycles - 1:  # Не ждем после последнего цикла
                    logger.info(f"\n💤 Sleeping {config.SCAN_INTERVAL_SECONDS}s...")
                    await asyncio.sleep(config.SCAN_INTERVAL_SECONDS)

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in cycle {cycle + 1}: {e}")
                await asyncio.sleep(10)

        logger.info("🏁 DRY RUN COMPLETED")

    def stop(self):
        self.running = False


if __name__ == "__main__":
    bot = TradingBotDryRun()

    try:
        # Запускаем 2 цикла для тестирования
        asyncio.run(bot.run(cycles=2))
    except KeyboardInterrupt:
        bot.stop()
        print("\n👋 Dry run stopped")