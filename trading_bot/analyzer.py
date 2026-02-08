import numpy as np
import pandas as pd
from decimal import Decimal
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum

from trading_bot.kernels import KernelRegression, KernelType
from trading_bot.data_manager import DataManager
from trading_bot.config import config
from trading_bot.utils import get_session_start_index
import logging

logger = logging.getLogger(__name__)


class DisplayMode(Enum):
    NET_RETURNS = "Net Returns"
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
    kernel_upper: float  # Верхняя полоса Deviation Band
    kernel_lower: float  # Нижняя полоса Deviation Band
    kernel_stdev: float  # Стандартное отклонение
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
        session_length: int = None,
        start_index: int = None,
        open_prices: np.ndarray = None,
        benchmark_open_prices: np.ndarray = None
    ) -> bool:
        """
        Проверяет, outperforms ли актив бенчмарк
        
        Args:
            asset_prices: Массив цен актива (hlc3).
            benchmark_prices: Массив цен бенчмарка.
            session_length: Длина сессии для Rolling window.
            start_index: Индекс начала сессии для Fixed window.
            open_prices: Массив цен открытия актива для расчета доходности первого бара.
            benchmark_open_prices: Массив цен открытия бенчмарка для расчета доходности первого бара.
        
        Returns:
            True = Green Zone (актив лучше бенчмарка)
            False = Red Zone (актив хуже бенчмарка)
        """
        if len(asset_prices) < 2 or len(benchmark_prices) < 2:
            return True # По умолчанию green
        
        # Определяем срез данных
        if start_index is not None:
            # Fixed Session
            asset_slice = asset_prices[start_index:]
            benchmark_slice = benchmark_prices[start_index:]
            
            # open_prices должен быть передан для корректного расчета первого бара
            if open_prices is not None:
                open_slice = open_prices[start_index:]
            else:
                # Fallback: если open_prices нет, используем asset_prices[0] как "предыдущее закрытие"
                # Это менее точно, но лучше, чем ничего
                open_slice = np.concatenate(([asset_prices[start_index-1] if start_index > 0 else asset_prices[0]], asset_slice[:-1]))
        else:
            # Rolling Session (используем последние N баров)
            length = session_length or len(asset_prices)
            length = min(length, len(asset_prices), len(benchmark_prices))
            
            asset_slice = asset_prices[-length:]
            benchmark_slice = benchmark_prices[-length:]
            
            # Для Rolling сессии используем стандартный расчет от предыдущего закрытия
            # (close - close[1])/close[1]
            open_slice = asset_slice[:-1] # Используем предыдущие цены закрытия как "open" для расчета
        
        # Расчёт доходностей
        # В Pine: assetReturn = sessionStart ? (close - open)/open : (close - close[1])/close[1]
        
        asset_returns = np.zeros(len(asset_slice))
        benchmark_returns = np.zeros(len(benchmark_slice))
        
        # Первый бар сессии: (close - open) / open
        # Примечание: open_prices[start_index] - это цена открытия ПЕРВОГО бара сессии
        if start_index is not None and open_prices is not None:
            asset_returns[0] = (asset_slice[0] - open_slice[0]) / open_slice[0]
            
            # DEBUG: Логируем расчет первого бара
            logger.debug(f"[FIRST_BAR] Asset: open={open_slice[0]:.6f}, close={asset_slice[0]:.6f}, return={asset_returns[0]:.6f}")
            
            # Для бенчмарка используем open бенчмарка, если доступен
            # Это соответствует логике Pine Script: request.security(benchmarkInput, '', assetReturn)
            if benchmark_open_prices is not None and len(benchmark_open_prices) > start_index:
                bench_open_slice = benchmark_open_prices[start_index:]
                benchmark_returns[0] = (benchmark_slice[0] - bench_open_slice[0]) / bench_open_slice[0]
                logger.debug(f"[FIRST_BAR] Benchmark: open={bench_open_slice[0]:.6f}, close={benchmark_slice[0]:.6f}, return={benchmark_returns[0]:.6f}")
            else:
                # Fallback: используем предыдущее закрытие бенчмарка
                logger.debug(f"[FIRST_BAR] Benchmark: NO open prices, using fallback")
                if len(benchmark_prices) > start_index + 1:
                    prev_bench_close = benchmark_prices[start_index - 1] if start_index > 0 else benchmark_slice[1]
                else:
                    prev_bench_close = benchmark_slice[-1] if len(benchmark_slice) > 1 else benchmark_slice[0]
                
                if prev_bench_close != 0:
                    benchmark_returns[0] = (benchmark_slice[0] - prev_bench_close) / prev_bench_close
                else:
                    benchmark_returns[0] = 0
                logger.debug(f"[FIRST_BAR] Benchmark (fallback): prev={prev_bench_close:.6f}, close={benchmark_slice[0]:.6f}, return={benchmark_returns[0]:.6f}")
        else:
            # Rolling session: стандартный расчет
            asset_returns[0] = 0
            if benchmark_open_prices is not None:
                bench_open_slice = benchmark_open_prices[-length:]
                benchmark_returns[0] = (benchmark_slice[0] - bench_open_slice[0]) / bench_open_slice[0]
            else:
                # Для TOTAL: используем rolling return между барами market cap
                if len(benchmark_prices) > len(benchmark_slice):
                    prev_bench_close = benchmark_prices[-length - 1]
                else:
                    prev_bench_close = benchmark_slice[1] if len(benchmark_slice) > 1 else benchmark_slice[0]
                
                if prev_bench_close != 0:
                    benchmark_returns[0] = (benchmark_slice[0] - prev_bench_close) / prev_bench_close
                else:
                    benchmark_returns[0] = 0

        # Остальные бары: (close - close[1])/close[1]
        asset_returns[1:] = (asset_slice[1:] - asset_slice[:-1]) / asset_slice[:-1]
        benchmark_returns[1:] = (benchmark_slice[1:] - benchmark_slice[:-1]) / benchmark_slice[:-1]
        
        # Цены для логирования (определяем здесь, чтобы использовать во всех режимах)
        open_price = asset_slice[0]
        close_price = asset_slice[-1]
        
        # DEBUG: Показываем какой режим используется
        logger.debug(f"[DISPLAY_MODE] {self.display_mode.value}")
        
        # DEBUG: Детали расчета
        logger.debug(f"[CALC_DETAILS] asset_returns_sum={np.sum(asset_returns):.8f}, asset_std={np.std(asset_returns, ddof=0):.8f}")
        logger.debug(f"[CALC_DETAILS] benchmark_returns_sum={np.sum(benchmark_returns):.8f}, benchmark_std={np.std(benchmark_returns, ddof=0):.8f}")
        
        if self.display_mode == DisplayMode.STANDARDIZED:
            # STANDARDIZED режим как в Pine Script:
            # 1. Берем стандартное отклонение актива
            # 2. Стандартизируем доходности бенчмарка
            # 3. Умножаем z-score на asset_std
            # 4. Сравниваем текущую цену с ожидаемой
            
            asset_std = np.std(asset_returns, ddof=0)
            if asset_std == 0:
                asset_std = 1
            
            mean_b = np.mean(benchmark_returns)
            std_b = np.std(benchmark_returns, ddof=0)
            if std_b == 0:
                standardized_benchmark = np.zeros_like(benchmark_returns)
            else:
                standardized_benchmark = (benchmark_returns - mean_b) / std_b
            
            # Кумулятивная доходность бенчмарка (стандартизированная * asset_std)
            cumulative_benchmark = np.sum(standardized_benchmark * asset_std)
            
            # Ожидаемая цена актива на основе бенчмарка
            expected_price = open_price * (1 + cumulative_benchmark)
            
            # Green если текущая цена >= ожидаемой
            is_green = close_price >= expected_price
            
            # DEBUG: Логируем для диагностики
            asset_return_pct = (close_price - open_price) / open_price * 100 if open_price > 0 else 0
            logger.debug(
                f"[ZONE_CHECK] Open={open_price:.4f} Close={close_price:.4f} "
                f"AssetReturn={asset_return_pct:.2f}% Expected={expected_price:.4f} "
                f"cum_bench_std={cumulative_benchmark:.6f} Result={'GREEN' if is_green else 'RED'}"
            )
            
            return is_green
        
        elif self.display_mode == DisplayMode.NORMALIZED:
            # Нормализованные доходности
            asset_std = np.std(asset_returns, ddof=0)
            benchmark_std = np.std(benchmark_returns, ddof=0)
            
            if benchmark_std == 0:
                ratio = 1
            else:
                ratio = asset_std / benchmark_std
            
            cumulative_asset = np.sum(asset_returns)
            cumulative_benchmark = np.sum(benchmark_returns * ratio)
            
            # Green если asset лучше бенчмарка
            is_green = cumulative_asset >= cumulative_benchmark
            
            asset_return_pct = (close_price - open_price) / open_price * 100 if open_price > 0 else 0
            logger.debug(
                f"[ZONE_CHECK] Open={open_price:.4f} Close={close_price:.4f} "
                f"AssetReturn={asset_return_pct:.2f}% cum_asset={cumulative_asset:.6f} "
                f"cum_bench={cumulative_benchmark:.6f} Result={'GREEN' if is_green else 'RED'}"
            )
            
            return is_green
        
        else:  # NET_RETURNS
            # Как в Pine Script: сравниваем текущую цену с ожидаемой
            cumulative_benchmark = np.sum(benchmark_returns)
            expected_price = open_price * (1 + cumulative_benchmark)
            
            # Green если текущая цена >= ожидаемой
            is_green = close_price >= expected_price
            
            asset_return_pct = (close_price - open_price) / open_price * 100 if open_price > 0 else 0
            logger.debug(
                f"[ZONE_CHECK] Open={open_price:.4f} Close={close_price:.4f} "
                f"AssetReturn={asset_return_pct:.2f}% Expected={expected_price:.4f} "
                f"cum_bench={cumulative_benchmark:.6f} Result={'GREEN' if is_green else 'RED'}"
            )
            
            return is_green
    
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
            bandwidth=config.BANDWIDTH,
            deviations=config.DEVIATIONS
        )
        
        # Relative Performance фильтр
        self.performance_filter = RelativePerformanceFilter(
            display_mode=DisplayMode[config.DISPLAY_MODE.upper().replace(" ", "_")]
        )
        
        # Кэш для benchmark данных
        self._benchmark_cache = {}
        self._total_market_cache = None
        self._total_market_timestamp = 0
    
    def get_benchmark_prices(self, length: int, asset_timestamps: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Получить цены бенчмарка (BTC или TOTAL от CoinGecko), выровненные по таймстемпам актива.
        
        Args:
            length: Максимальное количество баров для получения (если нет таймстемпов).
            asset_timestamps: Опционально np.array таймстемпов актива для выравнивания.
        
        Returns:
            Tuple[hlc3_prices, open_prices]: Массивы цен бенчмарка (hlc3 и open), выровненные по длине и времени.
        """
        cache_key_suffix = ""
        if asset_timestamps is not None:
            # Если переданы таймстемпы, создаем уникальный ключ кеша на основе первого и последнего таймстемпа
            cache_key_suffix = f"_{asset_timestamps[0]}_{asset_timestamps[-1]}"
            
        cache_key = f"{config.BENCHMARK_SYMBOL}_{length}{cache_key_suffix}"
        
        if cache_key in self._benchmark_cache:
            return self._benchmark_cache[cache_key]
        
        # Если бенчмарк = TOTAL, используем CoinGecko
        if config.BENCHMARK_SYMBOL.upper() == 'TOTAL':
            prices = self._get_total_benchmark(length, asset_timestamps)
            # Для TOTAL нет OHLC данных, возвращаем None для open цен
            open_prices = None
            self._benchmark_cache[cache_key] = (prices, open_prices)
            return prices, open_prices

        # Иначе используем Bybit как раньше
        # Получаем достаточно много данных, чтобы покрыть возможные пробелы
        # Для надежности берем немного больше, например, length + 50
        fetch_limit = min(length + 50, 1000) # Ограничиваем сверху
        df = self.dm.get_klines(config.BENCHMARK_SYMBOL, limit=fetch_limit)
        
        if df.empty:
            logger.warning(f"Could not fetch benchmark {config.BENCHMARK_SYMBOL}")
            return np.array([]), None
        
        # Используем hlc3 для бенчмарка
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        
        if asset_timestamps is not None and len(df) > 0:
            # Синхронизация по таймстемпам
            # df['timestamp'] - это datetimeIndex или Series
            
            # Создаем множество для быстрого поиска (O(1))
            # Хотя Pandas join/merge эффективнее для больших данных, здесь мы можем использовать merge
            
            # Преобразуем asset_timestamps в Series для merge
            asset_ts_series = pd.Series(asset_timestamps, name='timestamp')
            
            # Merge по timestamp
            # inner join гарантирует, что мы получим только совпадающие бары
            merged = pd.merge(
                asset_ts_series,
                df[['timestamp', 'hlc3', 'open']],
                on='timestamp',
                how='inner'
            )
            
            if len(merged) == 0:
                logger.warning(f"No matching benchmark data found for timestamps. Falling back to last {length} bars.")
                prices = df['hlc3'].values[-length:]
                open_prices = df['open'].values[-length:]
            else:
                # merged отсортирован по timestamp актива (так как asset_ts_series был отсортирован?)
                # asset_timestamps приходят из df, который сортируется в data_manager.
                # Поэтому merged должен быть отсортирован по порядку актива.
                prices = merged['hlc3'].values
                open_prices = merged['open'].values
        else:
            prices = df['hlc3'].values[-length:]
            open_prices = df['open'].values[-length:]
            
        self._benchmark_cache[cache_key] = (prices, open_prices)
        
        return prices, open_prices
    
    def _get_total_benchmark(self, length: int, asset_timestamps: np.ndarray = None) -> np.ndarray:
        """
        Получить исторические цены TOTAL (market cap) из CoinGecko.
        
        Args:
            length: Количество баров для получения
            asset_timestamps: Опционально np.array таймстемпов актива для выравнивания
            
        Returns:
            Массив цен бенчмарка, выровненный по длине и времени
        """
        import time
        
        # Определяем сколько дней истории нужно
        # Для 15-минутных свечей: 96 свечей в сутки
        # Берем с запасом для синхронизации
        if asset_timestamps is not None and len(asset_timestamps) > 0:
            # Рассчитываем диапазон времени в днях
            time_diff = asset_timestamps[-1] - asset_timestamps[0]
            
            # Проверяем тип: numpy.datetime64/timedelta64 или datetime
            if hasattr(time_diff, 'astype'):
                # numpy.datetime64 / numpy.timedelta64
                # Конвертируем в timedelta и берем days
                time_diff_td = pd.to_timedelta(time_diff)
                days_needed = max(7, int(time_diff_td.days) + 2)
            elif hasattr(time_diff, 'days'):
                # datetime.timedelta
                days_needed = max(7, int(time_diff.days) + 2)
            else:
                # numeric (milliseconds)
                days_needed = max(7, int(time_diff / (24 * 60 * 60 * 1000)) + 2)
            
            # Ограничиваем max дней для бесплатного CoinGecko API (365 дней)
            days_needed = min(days_needed, 365)
        else:
            # По умолчанию берем 7 дней
            days_needed = 7
        
        # Получаем исторические данные
        df_mc = self.dm.get_total_market_cap_history(days=days_needed)
        
        if df_mc is None or df_mc.empty:
            logger.warning("Could not fetch TOTAL market cap history from CoinGecko")
            return np.array([])
        
        if asset_timestamps is not None and len(asset_timestamps) > 0:
            # Конвертируем asset_timestamps в Unix timestamp (float) для избежания numpy datetime64 проблем
            asset_times_ms = pd.to_datetime(asset_timestamps).astype('int64') // 10**6  # в миллисекунды
            df_mc_times_ms = df_mc['timestamp'].astype('int64') // 10**6
            
            # Создаем словарь для быстрого lookup: timestamp_ms -> market_cap
            mc_dict = dict(zip(df_mc_times_ms, df_mc['market_cap'].values))
            
            # Для каждого asset timestamp находим ближайший MC timestamp
            prices = []
            for ts_ms in asset_times_ms:
                # Находим ближайший timestamp в MC данных
                if len(df_mc_times_ms) == 0:
                    prices.append(df_mc['market_cap'].iloc[-1] if len(df_mc) > 0 else 0)
                else:
                    # Ближайший timestamp (в миллисекундах)
                    closest_idx = np.abs(df_mc_times_ms - ts_ms).argmin()
                    prices.append(mc_dict[df_mc_times_ms[closest_idx]])
            
            prices = np.array(prices, dtype=float)
            logger.debug(f"TOTAL benchmark aligned: {len(prices)} points")
        else:
            # Без таймстемпов: возвращаем последние N значений
            if len(df_mc) >= length:
                prices = df_mc['market_cap'].values[-length:]
            else:
                # Дублируем первое значение если данных недостаточно
                prices = np.concatenate([
                    np.full(length - len(df_mc), df_mc['market_cap'].iloc[0]),
                    df_mc['market_cap'].values
                ])
        
        logger.info(f"TOTAL benchmark: {len(prices)} points, range=${prices.min():,.0f}-${prices.max():,.0f}")
        return prices
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Рассчитать все индикаторы с Deviation Bands"""
        df = df.copy()

        # Пользователь хочет использовать hlc3
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        prices = df['hlc3'].values

        # Kernel Regression с Deviation Bands
        kernel_ma, upper_band, lower_band = self.kernel.calculate_with_bands(prices)
        df['kernel_ma'] = kernel_ma
        df['kernel_upper'] = upper_band
        df['kernel_lower'] = lower_band
        df['kernel_stdev'] = (upper_band - kernel_ma) / config.DEVIATIONS  # Обратный расчёт для логирования
        
        # Направление (для определения тренда)
        df['kernel_direction'] = df['kernel_ma'].diff()
        
        # Crossover/Crossunder
        # В Pine: ta.crossover(nrp_sum, nrp_sum[1])
        df['kernel_cross_up'] = (df['kernel_ma'] > df['kernel_ma'].shift(1)) & (df['kernel_ma'].shift(1) <= df['kernel_ma'].shift(2))
        df['kernel_cross_down'] = (df['kernel_ma'] < df['kernel_ma'].shift(1)) & (df['kernel_ma'].shift(1) >= df['kernel_ma'].shift(2))
        
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
        # Передаем таймстемпы актива для синхронизации данных бенчмарка
        asset_timestamps = df['timestamp'].values
        benchmark_prices, benchmark_open_prices = self.get_benchmark_prices(len(df), asset_timestamps)
        
        if len(benchmark_prices) == 0:
            logger.warning(f"No benchmark data for {symbol}")
            return None
        
        # Пользователь хочет использовать hlc3 для всего, включая расчет зон
        asset_prices_for_zone = df['hlc3'].values
        open_prices = df['open'].values

        # Определяем зону (Green/Red)
        logger.debug(f"[ANALYZE] Checking zone for {symbol}. Asset len: {len(asset_prices_for_zone)}, Benchmark len: {len(benchmark_prices)}")
        
        # Определяем start_index для сессий
        start_index = None
        if config.SESSION_TYPE == 'Fixed':
            # Fixed сессия: используем заданные часы
            session_hours = config.SESSION_HOURS
            start_hour = config.SESSION_START_HOUR
            start_index = get_session_start_index(df, session_hours, start_hour)
            logger.debug(f"[SESSION] Fixed session. Start index: {start_index}, Time: {df.iloc[start_index]['timestamp']}")
        elif config.SESSION_TYPE == 'Exchange':
            # Exchange Session: начинаем с первого бара (00:00 UTC для крипты)
            timestamps = pd.to_datetime(df['timestamp'])
            
            # Начало текущего дня UTC
            from datetime import datetime
            now = datetime.utcnow()
            today_start = datetime(now.year, now.month, now.day, 0, 0, 0)
            
            # Ищем первый бар >= начала дня
            start_index = 0
            for i, ts in enumerate(timestamps):
                if ts >= today_start:
                    start_index = i
                    break
            
            logger.info(f"[SESSION_DEBUG] Now UTC: {now}, Today start: {today_start}")
            logger.info(f"[SESSION_DEBUG] First timestamp: {timestamps.iloc[0]}, Last: {timestamps.iloc[-1]}")
            logger.info(f"[SESSION] Exchange session. Start index: {start_index}, Time: {df.iloc[start_index]['timestamp']}")

        is_green_zone = self.performance_filter.is_outperforming(
            asset_prices_for_zone,
            benchmark_prices,
            session_length=config.SESSION_LENGTH,
            start_index=start_index,
            open_prices=open_prices,
            benchmark_open_prices=benchmark_open_prices
        )

        is_red_zone = not is_green_zone
        
        logger.info(f"[ZONE_STATUS] {symbol}: {'GREEN ZONE' if is_green_zone else 'RED ZONE'}")

        # Получаем последние значения
        last = df.iloc[-1]
        current_price = last['close']
        kernel_value = last['kernel_ma']
        kernel_upper = last['kernel_upper']
        kernel_lower = last['kernel_lower']
        kernel_stdev = last['kernel_stdev']
        
        # Проверяем сигналы
        cross_up = last['kernel_cross_up']
        cross_down = last['kernel_cross_down']
        
        signal = None
        
        # LONG: crossover + Green Zone
        if cross_up and is_green_zone:
            # Используем Decimal для точных расчетов
            current_price_dec = Decimal(str(current_price))
            sl_percent = Decimal(str(config.SL_PERCENT))
            tp_percent = Decimal(str(config.TP_PERCENT))
            
            stop_loss = float(current_price_dec * (Decimal('1') - sl_percent / Decimal('100')))
            take_profit = float(current_price_dec * (Decimal('1') + tp_percent / Decimal('100')))
            
            # Сила сигнала на основе относительной производительности
            perf_ratio = self.performance_filter.get_performance_ratio(
                asset_prices_for_zone, benchmark_prices
            )
            strength = min(abs(perf_ratio) * 10, 1.0)  # Нормализуем до 0-1
            
            signal = Signal(
                symbol=symbol,
                action='BUY',
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                kernel_value=kernel_value,
                kernel_upper=kernel_upper,
                kernel_lower=kernel_lower,
                kernel_stdev=kernel_stdev,
                is_outperforming=True,
                strength=strength,
                reason=f"Kernel crossover UP + Green Zone (outperforming by {perf_ratio*100:.2f}%)"
            )
        
        # SHORT: crossunder + Red Zone
        elif cross_down and is_red_zone:
            # Используем Decimal для точных расчетов
            current_price_dec = Decimal(str(current_price))
            sl_percent = Decimal(str(config.SL_PERCENT))
            tp_percent = Decimal(str(config.TP_PERCENT))
            
            stop_loss = float(current_price_dec * (Decimal('1') + sl_percent / Decimal('100')))
            take_profit = float(current_price_dec * (Decimal('1') - tp_percent / Decimal('100')))
            
            perf_ratio = self.performance_filter.get_performance_ratio(
                asset_prices_for_zone, benchmark_prices
            )
            strength = min(abs(perf_ratio) * 10, 1.0)
            
            signal = Signal(
                symbol=symbol,
                action='SELL',
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                kernel_value=kernel_value,
                kernel_upper=kernel_upper,
                kernel_lower=kernel_lower,
                kernel_stdev=kernel_stdev,
                is_outperforming=False,
                strength=strength,
                reason=f"Kernel crossover DOWN + Red Zone (underperforming by {abs(perf_ratio)*100:.2f}%)"
            )
        
        if signal:
            logger.info(
                f"📊 {signal.action} signal for {symbol}: "
                f"price={current_price:.6f}, "
                f"zone={'GREEN' if signal.is_outperforming else 'RED'}, "
                f"strength={signal.strength:.2f}, "
                f"kernel={signal.kernel_value:.6f}, "
                f"bands=[{signal.kernel_lower:.6f}, {signal.kernel_upper:.6f}]"
            )
        
        return signal
    
    def get_zone_status(self, symbol: str) -> Tuple[bool, float]:
        """
        Получить текущий статус зоны для символа
        Returns: (is_green_zone, performance_ratio)
        """
        df = self.dm.get_klines(symbol, limit=config.KLINES_LIMIT)
        
        if df.empty:
            return True, 0.0
            
        asset_timestamps = df['timestamp'].values
        benchmark_prices, benchmark_open_prices = self.get_benchmark_prices(len(df), asset_timestamps)

        if len(benchmark_prices) == 0:
            return True, 0.0

        # Используем hlc3 для расчета зон
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        asset_prices = df['hlc3'].values
        open_prices = df['open'].values
        
        start_index = None
        if config.SESSION_TYPE == 'Fixed':
            start_index = get_session_start_index(df, config.SESSION_HOURS, config.SESSION_START_HOUR)
        
        is_green = self.performance_filter.is_outperforming(
            asset_prices, benchmark_prices, config.SESSION_LENGTH, start_index, open_prices, benchmark_open_prices
        )
        
        perf_ratio = self.performance_filter.get_performance_ratio(
            asset_prices, benchmark_prices
        )
        
        return is_green, perf_ratio