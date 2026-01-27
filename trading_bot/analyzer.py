import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum

from trading_bot.kernels import KernelRegression, KernelType
from trading_bot.data_manager import DataManager
from trading_bot.config import config
import logging

logger = logging.getLogger(__name__)


class DisplayMode(Enum):
    NET_RETURN = "Net Returns"
    NORMALIZED = "Rescaled Returns"
    STANDARDIZED = "Standardized Returns"


@dataclass
class Signal:
    symbol: str
    action: str  # 'BUY' or 'SELL'
    price: float
    stop_loss: float
    take_profit: float
    kernel_value: float
    is_outperforming: bool
    strength: float
    reason: str


class RelativePerformanceFilter:
    """Фильтр относительной производительности (сравнение с бенчмарком)"""
    
    def __init__(self, display_mode: DisplayMode = DisplayMode.STANDARDIZED):
        self.display_mode = display_mode
    
    def calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """Расчёт доходностей"""
        returns = np.zeros(len(prices))
        returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        return returns
    
    def standardize(self, returns: np.ndarray) -> np.ndarray:
        """Стандартизация доходностей (z-score)"""
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return np.zeros_like(returns)
        return (returns - mean) / std
    
    def is_outperforming(
        self, 
        asset_prices: np.ndarray, 
        benchmark_prices: np.ndarray,
        session_length: int = None
    ) -> bool:
        """
        Проверяет, outperforms ли актив бенчмарк
        
        Returns:
            True = Green Zone (актив лучше бенчмарка)
            False = Red Zone (актив хуже бенчмарка)
        """
        if len(asset_prices) < 2 or len(benchmark_prices) < 2:
            return True  # По умолчанию green
        
        # Используем session_length или всю длину
        length = session_length or len(asset_prices)
        
        asset_slice = asset_prices[-length:]
        benchmark_slice = benchmark_prices[-length:]
        
        # Расчёт доходностей
        asset_returns = self.calculate_returns(asset_slice)
        benchmark_returns = self.calculate_returns(benchmark_slice)
        
        if self.display_mode == DisplayMode.STANDARDIZED:
            # Стандартизированные доходности
            asset_std = np.std(asset_returns)
            if asset_std == 0:
                asset_std = 1
            
            standardized_benchmark = self.standardize(benchmark_returns)
            cumulative_benchmark = np.sum(standardized_benchmark * asset_std)
        
        elif self.display_mode == DisplayMode.NORMALIZED:
            # Нормализованные доходности
            asset_std = np.std(asset_returns)
            benchmark_std = np.std(benchmark_returns)
            
            if benchmark_std == 0:
                ratio = 1
            else:
                ratio = asset_std / benchmark_std
            
            cumulative_benchmark = np.sum(benchmark_returns * ratio)
        
        else:  # NET_RETURN
            cumulative_benchmark = np.sum(benchmark_returns)
        
        # Сравнение: текущая цена vs ожидаемая на основе бенчмарка
        open_price = asset_slice[0]
        close_price = asset_slice[-1]
        expected_price = open_price * (1 + cumulative_benchmark)
        
        return close_price >= expected_price
    
    def get_performance_ratio(
        self, 
        asset_prices: np.ndarray, 
        benchmark_prices: np.ndarray
    ) -> float:
        """Получить коэффициент относительной производительности"""
        if len(asset_prices) < 2:
            return 0.0
        
        asset_return = (asset_prices[-1] - asset_prices[0]) / asset_prices[0]
        benchmark_return = (benchmark_prices[-1] - benchmark_prices[0]) / benchmark_prices[0]
        
        return asset_return - benchmark_return


class MultiKernelAnalyzer:
    """
    Полная реализация Multi Kernel Regression + Relative Performance
    """
    
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager
        
        # Kernel Regression параметры
        self.kernel = KernelRegression(
            kernel_type=KernelType[config.KERNEL_TYPE.upper().replace(" ", "_")],
            bandwidth=config.BANDWIDTH
        )
        
        # Relative Performance фильтр
        self.performance_filter = RelativePerformanceFilter(
            display_mode=DisplayMode[config.DISPLAY_MODE.upper().replace(" ", "_")]
        )
        
        # Кэш для benchmark данных
        self._benchmark_cache = {}
        self._total_market_cache = None
        self._total_market_timestamp = 0
    
    def get_benchmark_prices(self, length: int) -> np.ndarray:
        """Получить цены бенчмарка (BTC или TOTAL от CoinGecko)"""
        cache_key = f"{config.BENCHMARK_SYMBOL}_{length}"
        
        if cache_key in self._benchmark_cache:
            return self._benchmark_cache[cache_key]
        
        # Если бенчмарк = TOTAL, используем CoinGecko
        if config.BENCHMARK_SYMBOL.upper() == 'TOTAL':
            return self._get_total_benchmark(length)
        
        # Иначе используем Bybit как раньше
        df = self.dm.get_klines(config.BENCHMARK_SYMBOL, limit=length)
        
        if df.empty:
            logger.warning(f"Could not fetch benchmark {config.BENCHMARK_SYMBOL}")
            return np.array([])
        
        prices = df['close'].values
        self._benchmark_cache[cache_key] = prices
        
        return prices
    
    def _get_total_benchmark(self, length: int) -> np.ndarray:
        """Получить псевдо-цены TOTAL из CoinGecko (market cap)"""
        import time
        current_time = time.time()
        
        # Проверяем кеш (5 минут)
        if self._total_market_cache is not None and \
           (current_time - self._total_market_timestamp < 300):
            # Создаем массив цен на основе кешированных данных
            # Для упрощения используем текущий market cap как "цену"
            market_cap = self._total_market_cache
            return np.full(length, market_cap)
        
        # Получаем данные с CoinGecko
        data = self.dm.get_total_market_data()
        
        if data and data.get('total_market_cap'):
            self._total_market_cache = data['total_market_cap']
            self._total_market_timestamp = current_time
            
            # Создаем массив цен (используем market cap как "цену")
            prices = np.full(length, data['total_market_cap'])
            
            logger.info(f"TOTAL benchmark from CoinGecko: ${data['total_market_cap']:,.0f}")
            return prices
        
        logger.warning("Could not fetch TOTAL benchmark from CoinGecko")
        return np.array([])
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Рассчитать все индикаторы"""
        df = df.copy()

        # Используем hlc3 как источник данных (high + low + close) / 3
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        prices = df['hlc3'].values

        # Kernel Regression
        df['kernel_ma'] = self.kernel.calculate_series(prices)
        
        # Направление (для определения тренда)
        df['kernel_direction'] = df['kernel_ma'].diff()
        
        # Crossover/Crossunder
        df['kernel_cross_up'] = (df['kernel_direction'] > 0) & (df['kernel_direction'].shift(1) <= 0)
        df['kernel_cross_down'] = (df['kernel_direction'] < 0) & (df['kernel_direction'].shift(1) >= 0)
        
        return df
    
    def analyze(self, symbol: str) -> Optional[Signal]:
        """
        Анализ символа и генерация сигнала
        Полная логика индикатора
        """
        # Получаем данные актива
        df = self.dm.get_klines(symbol, limit=config.KLINES_LIMIT)
        
        if df.empty or len(df) < config.BANDWIDTH + 10:
            logger.warning(f"Not enough data for {symbol}")
            return None
        
        # Рассчитываем индикаторы
        df = self.calculate_indicators(df)
        
        # Получаем данные бенчмарка
        benchmark_prices = self.get_benchmark_prices(len(df))
        
        if len(benchmark_prices) == 0:
            logger.warning(f"No benchmark data for {symbol}")
            return None
        
        # Используем hlc3 для расчета зон и сигналов
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        asset_prices = df['hlc3'].values

        # Определяем зону (Green/Red)
        is_green_zone = self.performance_filter.is_outperforming(
            asset_prices,
            benchmark_prices,
            session_length=config.SESSION_LENGTH
        )

        is_red_zone = not is_green_zone

        # Получаем последние значения
        last = df.iloc[-1]
        current_price = last['close']
        kernel_value = last['kernel_ma']
        
        # Проверяем сигналы
        cross_up = last['kernel_cross_up']
        cross_down = last['kernel_cross_down']
        
        signal = None
        
        # LONG: crossover + Green Zone
        if cross_up and is_green_zone:
            stop_loss = current_price * (1 - config.SL_PERCENT / 100)
            take_profit = current_price * (1 + config.TP_PERCENT / 100)
            
            # Сила сигнала на основе относительной производительности
            perf_ratio = self.performance_filter.get_performance_ratio(
                asset_prices, benchmark_prices
            )
            strength = min(abs(perf_ratio) * 10, 1.0)  # Нормализуем до 0-1
            
            signal = Signal(
                symbol=symbol,
                action='BUY',
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                kernel_value=kernel_value,
                is_outperforming=True,
                strength=strength,
                reason=f"Kernel crossover UP + Green Zone (outperforming by {perf_ratio*100:.2f}%)"
            )
        
        # SHORT: crossunder + Red Zone  
        elif cross_down and is_red_zone:
            stop_loss = current_price * (1 + config.SL_PERCENT / 100)
            take_profit = current_price * (1 - config.TP_PERCENT / 100)
            
            perf_ratio = self.performance_filter.get_performance_ratio(
                asset_prices, benchmark_prices
            )
            strength = min(abs(perf_ratio) * 10, 1.0)
            
            signal = Signal(
                symbol=symbol,
                action='SELL',
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                kernel_value=kernel_value,
                is_outperforming=False,
                strength=strength,
                reason=f"Kernel crossover DOWN + Red Zone (underperforming by {abs(perf_ratio)*100:.2f}%)"
            )
        
        if signal:
            logger.info(
                f"📊 {signal.action} signal for {symbol}: "
                f"price={current_price:.6f}, "
                f"zone={'GREEN' if signal.is_outperforming else 'RED'}, "
                f"strength={signal.strength:.2f}"
            )
        
        return signal
    
    def get_zone_status(self, symbol: str) -> Tuple[bool, float]:
        """
        Получить текущий статус зоны для символа
        Returns: (is_green_zone, performance_ratio)
        """
        df = self.dm.get_klines(symbol, limit=config.KLINES_LIMIT)
        benchmark_prices = self.get_benchmark_prices(len(df))

        if df.empty or len(benchmark_prices) == 0:
            return True, 0.0

        # Используем hlc3 для расчета зон
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        asset_prices = df['hlc3'].values
        
        is_green = self.performance_filter.is_outperforming(
            asset_prices, benchmark_prices, config.SESSION_LENGTH
        )
        
        perf_ratio = self.performance_filter.get_performance_ratio(
            asset_prices, benchmark_prices
        )
        
        return is_green, perf_ratio