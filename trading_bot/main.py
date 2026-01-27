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


class TradingBot:
    def __init__(self):
        self.dm = DataManager()
        self.screener = Screener(self.dm)
        self.analyzer = MultiKernelAnalyzer(self.dm)
        self.executor = Executor(self.dm)
        self.running = False
    
    async def run_cycle(self):
        """Один цикл сканирования"""
        cycle_start = time.time()
        
        logger.info("=" * 60)
        logger.info(f"🔄 SCAN CYCLE START | {datetime.now()}")
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
        
        # 4. Исполняем топ сигналы
        executed = 0
        for signal in signals[:2]:  # Максимум 2 сделки за цикл
            if self.executor.execute(signal):
                executed += 1
            await asyncio.sleep(0.5)
        
        # 5. Мониторинг позиций
        await self.monitor_positions()
        
        # Очищаем кеш бенчмарка
        self.analyzer._benchmark_cache.clear()
        self.dm.clear_cache()
        
        cycle_time = time.time() - cycle_start
        logger.info(f"\n⏱️ Cycle completed in {cycle_time:.2f}s | Executed: {executed}")
    
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
    
    async def run(self):
        """Основной цикл"""
        self.running = True
        
        logger.info("🚀 TRADING BOT STARTED")
        logger.info(f"   Kernel: {config.KERNEL_TYPE} (bandwidth={config.BANDWIDTH})")
        logger.info(f"   Benchmark: {config.BENCHMARK_SYMBOL}")
        logger.info(f"   Timeframe: {config.TIMEFRAME}m")
        logger.info(f"   SL: {config.SL_PERCENT}% | TP: {config.TP_PERCENT}%")
        logger.info(f"   Leverage: {config.LEVERAGE}x")
        
        while self.running:
            try:
                await self.run_cycle()
                
                logger.info(f"\n💤 Sleeping {config.SCAN_INTERVAL_SECONDS}s...")
                await asyncio.sleep(config.SCAN_INTERVAL_SECONDS)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self.running = False
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(10)
    
    def stop(self):
        self.running = False


if __name__ == "__main__":
    bot = TradingBot()
    
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        bot.stop()
        print("\n👋 Bot stopped")