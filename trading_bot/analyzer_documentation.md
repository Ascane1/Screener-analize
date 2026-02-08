# Документация модуля analyzer.py

## Обзор

Модуль `analyzer.py` является ядром торгового бота, отвечающим за технический анализ криптовалютных активов. Он реализует стратегию на основе **Kernel Regression** с **Deviation Bands** и фильтром **Relative Performance** для определения торговых сигналов.

## Архитектура модуля

```
┌─────────────────────────────────────────────────────────────────┐
│                        analyzer.py                              │
├─────────────────────────────────────────────────────────────────┤
│  Enums:          DisplayMode                                    │
│  Data Classes:   Signal                                         │
│  Classes:        RelativePerformanceFilter                      │
│                  MultiKernelAnalyzer                            │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  kernels.py   │    │ data_manager  │    │    config     │
│KernelRegression│    │   .py         │    │   .py         │
└───────────────┘    └───────────────┘    └───────────────┘
```

---

## Импорты и зависимости

```python
import numpy as np              # Численные вычисления, работа с массивами
import pandas as pd             # Работа с DataFrame, временными рядами
from decimal import Decimal     # Точные десятичные вычисления (SL/TP)
from dataclasses import dataclass  # Декоратор для класса Signal
from typing import Optional, Tuple, List  # Типизация
from enum import Enum           # Перечисления

from trading_bot.kernels import KernelRegression, KernelType  # Ядро регрессии
from trading_bot.data_manager import DataManager               # Управление данными
from trading_bot.config import config                          # Конфигурация
from trading_bot.utils import get_session_start_index          # Утилиты
import logging
```

---

## Enum: DisplayMode

```python
class DisplayMode(Enum):
    NET_RETURNS = "Net Returns"           # Чистая доходность
    NORMALIZED = "Rescaled Returns"       # Нормализованная доходность
    STANDARDIZED = "Standardized Returns" # Стандартизированная (z-score)
```

**Назначение**: Определяет метод расчета относительной производительности актива относительно бенчмарка.

**Источник данных**: Значение берется из `config.DISPLAY_MODE` и конвертируется в Enum.

---

## Data Class: Signal

```python
@dataclass
class Signal:
    symbol: str            # Торговая пара (например, "BTCUSDT")
    action: str            # Действие: 'BUY' или 'SELL'
    price: float           # Текущая цена при генерации сигнала
    stop_loss: float       # Цена стоп-лосса
    take_profit: float     # Цена тейк-профита
    kernel_value: float    # Значение линии ядра регрессии
    kernel_upper: float    # Верхняя полоса отклонения
    kernel_lower: float    # Нижняя полоса отклонения
    kernel_stdev: float    # Стандартное отклонение
    is_outperforming: bool # True = Green Zone, False = Red Zone
    strength: float        # Сила сигнала (0.0 - 1.0)
    reason: str            # Описание причины сигнала
```

**Назначение**: Структура данных для хранения информации о торговом сигнале.

**Куда передается**: Возвращается методом `analyze()`, передается в `executor.py` для исполнения сделок.

---

## Class: RelativePerformanceFilter

Класс для фильтрации активов на основе их производительности относительно бенчмарка (BTC или TOTAL market cap).

### Constructor: `__init__`

```python
def __init__(self, display_mode: DisplayMode = DisplayMode.STANDARDIZED):
    self.display_mode = display_mode
```

**Входные параметры**:
- `display_mode`: Режим расчета производительности (по умолчанию STANDARDIZED)

---

### Method: `calculate_returns`

```python
def calculate_returns(self, prices: np.ndarray) -> np.ndarray:
    """Расчёт доходностей"""
    returns = np.zeros(len(prices))
    returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    return returns
```

**Назначение**: Вычисляет процентные доходности из массива цен.

**Входные данные**:
- `prices`: np.ndarray - массив цен (обычно hlc3)

**Логика работы**:
1. Создает массив нулей той же длины
2. Для каждого бара (кроме первого): `(close - prev_close) / prev_close`
3. Первый бар всегда имеет доходность 0

**Выходные данные**:
- np.ndarray - массив доходностей

---

### Method: `standardize`

```python
def standardize(self, returns: np.ndarray) -> np.ndarray:
    """Стандартизация доходностей (z-score)"""
    mean = np.mean(returns)
    std = np.std(returns)
    if std == 0:
        return np.zeros_like(returns)
    return (returns - mean) / std
```

**Назначение**: Преобразует доходности в z-scores (стандартные отклонения от среднего).

**Входные данные**:
- `returns`: np.ndarray - массив доходностей

**Логика работы**:
1. Вычисляет среднее значение доходностей
2. Вычисляет стандартное отклонение
3. Если std = 0, возвращает массив нулей
4. Иначе: `(return - mean) / std`

**Выходные данные**:
- np.ndarray - стандартизированные доходности

---

### Method: `is_outperforming` (Core Method)

```python
def is_outperforming(
    self,
    asset_prices: np.ndarray,           # Цены актива (hlc3)
    benchmark_prices: np.ndarray,       # Цены бенчмарка
    session_length: int = None,         # Длина Rolling окна
    start_index: int = None,            # Индекс начала Fixed сессии
    open_prices: np.ndarray = None,     # Цены открытия актива
    benchmark_open_prices: np.ndarray = None  # Цены открытия бенчмарка
) -> bool
```

**Назначение**: Определяет, outperforming ли актив бенчмарк (Green Zone) или underperforming (Red Zone).

**Источники данных**:
- `asset_prices`: Из `df['hlc3']` - средние цены актива (high+low+close)/3
- `benchmark_prices`: Из `get_benchmark_prices()` - цены бенчмарка (BTC или TOTAL)
- `open_prices`: Из `df['open']` - цены открытия
- `benchmark_open_prices`: Из `get_benchmark_prices()`

**Логика работы**:

#### 1. Определение среза данных

**Fixed Session** (когда `start_index` передан):
- Используются данные от `start_index` до конца
- Первый бар сессии рассчитывается как `(close - open) / open`
- Остальные бары: `(close - prev_close) / prev_close`

**Rolling Session** (когда `start_index` is None):
- Используются последние `session_length` баров
- Расчет по стандартной формуле: `(close - prev_close) / prev_close`

#### 2. Расчет доходностей

```
Первый бар (Fixed Session):  (close[0] - open[0]) / open[0]
Остальные бары:              (close[i] - close[i-1]) / close[i-1]
```

#### 3. Режимы сравнения

**STANDARDIZED Mode** (как в Pine Script):
```python
asset_std = np.std(asset_returns)
standardized_benchmark = (benchmark_returns - mean_b) / std_b
cumulative_benchmark = np.sum(standardized_benchmark * asset_std)
expected_price = open_price * (1 + cumulative_benchmark)
is_green = close_price >= expected_price
```

**Логика**: Сравнивает фактическую цену актива с ожидаемой ценой, рассчитанной на основе стандартизированной доходности бенчмарка.

**NORMALIZED Mode**:
```python
ratio = asset_std / benchmark_std
cumulative_asset = np.sum(asset_returns)
cumulative_benchmark = np.sum(benchmark_returns * ratio)
is_green = cumulative_asset >= cumulative_benchmark
```

**NET_RETURNS Mode**:
```python
cumulative_benchmark = np.sum(benchmark_returns)
expected_price = open_price * (1 + cumulative_benchmark)
is_green = close_price >= expected_price
```

**Выходные данные**:
- `bool`: True = Green Zone (актив лучше бенчмарка), False = Red Zone

---

### Method: `get_performance_ratio`

```python
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
```

**Назначение**: Вычисляет разницу в доходности между активом и бенчмарком.

**Использование**: Для расчета `strength` сигнала в методе `analyze()`.

---

## Class: MultiKernelAnalyzer

Основной класс анализатора, объединяющий Kernel Regression и Relative Performance фильтр.

### Constructor: `__init__`

```python
def __init__(self, data_manager: DataManager):
    self.dm = data_manager
    
    # Инициализация Kernel Regression из конфигурации
    self.kernel = KernelRegression(
        kernel_type=KernelType[config.KERNEL_TYPE.upper().replace(" ", "_")],
        bandwidth=config.BANDWIDTH,
        deviations=config.DEVIATIONS
    )
    
    # Инициализация фильтра производительности
    self.performance_filter = RelativePerformanceFilter(
        display_mode=DisplayMode[config.DISPLAY_MODE.upper().replace(" ", "_")]
    )
    
    # Кэширование данных бенчмарка
    self._benchmark_cache = {}
    self._total_market_cache = None
    self._total_market_timestamp = 0
```

**Входные параметры**:
- `data_manager`: Экземпляр `DataManager` для получения рыночных данных

**Источники конфигурации**:
- `config.KERNEL_TYPE`: Тип ядра (Gaussian, Epanechnikov, и т.д.)
- `config.BANDWIDTH`: Ширина окна ядра
- `config.DEVIATIONS`: Множитель для полос отклонения
- `config.DISPLAY_MODE`: Режим отображения

---

### Method: `get_benchmark_prices`

```python
def get_benchmark_prices(
    self, 
    length: int, 
    asset_timestamps: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]
```

**Назначение**: Получает цены бенчмарка (BTC или TOTAL), синхронизированные с временными рядами актива.

**Источники данных**:
- Если `config.BENCHMARK_SYMBOL == 'TOTAL'`: CoinGecko API (через `_get_total_benchmark()`)
- Иначе: Bybit API (через `data_manager.get_klines()`)

**Логика работы**:

1. **Проверка кэша**:
   ```python
   cache_key = f"{config.BENCHMARK_SYMBOL}_{length}{cache_key_suffix}"
   if cache_key in self._benchmark_cache:
       return self._benchmark_cache[cache_key]
   ```

2. **TOTAL Benchmark (CoinGecko)**:
   - Вызывает `_get_total_benchmark()`
   - Возвращает `hlc3` цены и `None` для open цен

3. **Bybit Benchmark (BTC)**:
   - Получает данные: `self.dm.get_klines(config.BENCHMARK_SYMBOL, limit=fetch_limit)`
   - Рассчитывает hlc3: `(high + low + close) / 3`
   - **Синхронизация по таймстемпам**:
     ```python
     merged = pd.merge(
         asset_ts_series,
         df[['timestamp', 'hlc3', 'open']],
         on='timestamp',
         how='inner'
     )
     ```
   - Использует inner join для получения только общих баров

**Выходные данные**:
- `Tuple[hlc3_prices, open_prices]`: Кортеж массивов цен

**Куда передаются**: В `is_outperforming()` для определения зоны.

---

### Method: `_get_total_benchmark` (Private)

```python
def _get_total_benchmark(
    self, 
    length: int, 
    asset_timestamps: np.ndarray = None
) -> np.ndarray
```

**Назначение**: Получает исторические данные TOTAL market cap из CoinGecko.

**Источник данных**: `self.dm.get_total_market_cap_history(days=days_needed)`

**Логика работы**:

1. **Расчет необходимого количества дней**:
   ```python
   time_diff = asset_timestamps[-1] - asset_timestamps[0]
   days_needed = max(7, int(time_diff.days) + 2)
   days_needed = min(days_needed, 365)  # Ограничение API
   ```

2. **Синхронизация по времени**:
   ```python
   asset_times_ms = pd.to_datetime(asset_timestamps).astype('int64') // 10**6
   df_mc_times_ms = df_mc['timestamp'].astype('int64') // 10**6
   
   # Для каждого таймстемпа актива находим ближайший в MC данных
   for ts_ms in asset_times_ms:
       closest_idx = np.abs(df_mc_times_ms - ts_ms).argmin()
       prices.append(mc_dict[df_mc_times_ms[closest_idx]])
   ```

**Выходные данные**:
- np.ndarray - массив значений market cap, выровненный по времени актива

---

### Method: `calculate_indicators`

```python
def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame
```

**Назначение**: Рассчитывает все технические индикаторы на основе Kernel Regression.

**Входные данные**:
- `df`: pd.DataFrame с колонками 'open', 'high', 'low', 'close'

**Источники данных**:
- Цены OHLC из DataFrame
- Параметры ядра из `self.kernel`

**Логика работы**:

1. **Расчет hlc3**:
   ```python
   df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
   ```

2. **Kernel Regression с полосами**:
   ```python
   kernel_ma, upper_band, lower_band = self.kernel.calculate_with_bands(prices)
   df['kernel_ma'] = kernel_ma
   df['kernel_upper'] = upper_band
   df['kernel_lower'] = lower_band
   df['kernel_stdev'] = (upper_band - kernel_ma) / config.DEVIATIONS
   ```

3. **Определение направления**:
   ```python
   df['kernel_direction'] = df['kernel_ma'].diff()
   ```

4. **Сигналы пересечения**:
   ```python
   # Crossover: текущее > предыдущего И предыдущее <= позапрошлого
   df['kernel_cross_up'] = (df['kernel_ma'] > df['kernel_ma'].shift(1)) & 
                           (df['kernel_ma'].shift(1) <= df['kernel_ma'].shift(2))
   
   # Crossunder: текущее < предыдущего И предыдущее >= позапрошлого
   df['kernel_cross_down'] = (df['kernel_ma'] < df['kernel_ma'].shift(1)) & 
                             (df['kernel_ma'].shift(1) >= df['kernel_ma'].shift(2))
   ```

**Выходные данные**:
- pd.DataFrame с добавленными колонками индикаторов

**Куда передается**: Используется в методе `analyze()`.

---

### Method: `analyze` (Core Method)

```python
def analyze(self, symbol: str) -> Optional[Signal]
```

**Назначение**: Основной метод анализа актива и генерации торгового сигнала.

**Входные параметры**:
- `symbol`: str - торговая пара (например, "BTCUSDT")

**Источники данных**:
1. `self.dm.get_klines(symbol, limit=config.KLINES_LIMIT)` - данные актива
2. `self.get_benchmark_prices()` - данные бенчмарка
3. `config.SESSION_TYPE` - тип сессии (Fixed/Exchange/Rolling)
4. `config.SL_PERCENT`, `config.TP_PERCENT` - уровни стоп-лосса и тейк-профита

**Логика работы** (пошагово):

#### Шаг 1: Получение данных актива
```python
df = self.dm.get_klines(symbol, limit=config.KLINES_LIMIT)
if df.empty or len(df) < config.BANDWIDTH + 10:
    return None  # Недостаточно данных
```

#### Шаг 2: Расчет индикаторов
```python
df = self.calculate_indicators(df)
```

#### Шаг 3: Получение данных бенчмарка
```python
asset_timestamps = df['timestamp'].values
benchmark_prices, benchmark_open_prices = self.get_benchmark_prices(len(df), asset_timestamps)
```

#### Шаг 4: Определение типа сессии и start_index

**Fixed Session**:
```python
if config.SESSION_TYPE == 'Fixed':
    session_hours = config.SESSION_HOURS
    start_hour = config.SESSION_START_HOUR
    start_index = get_session_start_index(df, session_hours, start_hour)
```

**Exchange Session** (начало дня UTC):
```python
elif config.SESSION_TYPE == 'Exchange':
    now = datetime.utcnow()
    today_start = datetime(now.year, now.month, now.day, 0, 0, 0)
    # Ищем первый бар >= начала дня
```

**Rolling Session**: `start_index = None`

#### Шаг 5: Определение зоны (Green/Red)
```python
is_green_zone = self.performance_filter.is_outperforming(
    asset_prices_for_zone,      # hlc3 цены актива
    benchmark_prices,            # цены бенчмарка
    session_length=config.SESSION_LENGTH,
    start_index=start_index,
    open_prices=open_prices,
    benchmark_open_prices=benchmark_open_prices
)
is_red_zone = not is_green_zone
```

#### Шаг 6: Получение последних значений
```python
last = df.iloc[-1]
current_price = last['close']
kernel_value = last['kernel_ma']
kernel_upper = last['kernel_upper']
kernel_lower = last['kernel_lower']
cross_up = last['kernel_cross_up']
cross_down = last['kernel_cross_down']
```

#### Шаг 7: Генерация сигнала

**LONG сигнал** (Crossover + Green Zone):
```python
if cross_up and is_green_zone:
    # Расчет SL/TP с использованием Decimal для точности
    current_price_dec = Decimal(str(current_price))
    sl_percent = Decimal(str(config.SL_PERCENT))
    tp_percent = Decimal(str(config.TP_PERCENT))
    
    stop_loss = float(current_price_dec * (Decimal('1') - sl_percent / Decimal('100')))
    take_profit = float(current_price_dec * (Decimal('1') + tp_percent / Decimal('100')))
    
    # Расчет силы сигнала
    perf_ratio = self.performance_filter.get_performance_ratio(asset_prices_for_zone, benchmark_prices)
    strength = min(abs(perf_ratio) * 10, 1.0)
    
    signal = Signal(
        symbol=symbol,
        action='BUY',
        price=current_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        ...
        reason=f"Kernel crossover UP + Green Zone (outperforming by {perf_ratio*100:.2f}%)"
    )
```

**SHORT сигнал** (Crossunder + Red Zone):
```python
elif cross_down and is_red_zone:
    stop_loss = float(current_price_dec * (Decimal('1') + sl_percent / Decimal('100')))
    take_profit = float(current_price_dec * (Decimal('1') - tp_percent / Decimal('100')))
    ...
    action='SELL'
```

**Выходные данные**:
- `Signal` объект если сигнал сгенерирован
- `None` если условия не выполнены

**Куда передается**: В `screener.py` для массового анализа или в `main.py` для торговли.

---

### Method: `get_zone_status`

```python
def get_zone_status(self, symbol: str) -> Tuple[bool, float]
```

**Назначение**: Получает текущий статус зоны для символа без генерации сигнала.

**Входные параметры**:
- `symbol`: str - торговая пара

**Источники данных**: Те же, что и в `analyze()`

**Логика работы**:
1. Получает данные актива
2. Получает данные бенчмарка
3. Рассчитывает hlc3
4. Определяет зону через `is_outperforming()`
5. Рассчитывает `performance_ratio`

**Выходные данные**:
- `Tuple[is_green_zone, performance_ratio]`

**Использование**: Для мониторинга состояния без генерации торговых сигналов.

---

## Поток данных

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ПОТОК ДАННЫХ В ANALYZE()                        │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐         ┌──────────────┐         ┌──────────────────┐
  │  DataManager │         │   Bybit API  │         │   CoinGecko API  │
  │   .py        │◄────────│   (BTC)      │         │   (TOTAL MC)     │
  └──────┬───────┘         └──────────────┘         └──────────────────┘
         │
         │ get_klines(symbol)
         ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  1. Загрузка данных актива                                       │
  │     - OHLC свечи                                                 │
  │     - Timestamps                                                 │
  └──────────────────────────────────────────────────────────────────┘
         │
         │ calculate_indicators()
         ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  2. Расчет индикаторов                                           │
  │     - hlc3 = (high + low + close) / 3                           │
  │     - kernel_ma (ядро регрессии)                                 │
  │     - upper/lower bands                                          │
  │     - cross_up / cross_down                                      │
  └──────────────────────────────────────────────────────────────────┘
         │
         │ get_benchmark_prices()
         ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  3. Загрузка бенчмарка                                           │
  │     - Если TOTAL: CoinGecko market cap                          │
  │     - Иначе: Bybit (BTC)                                         │
  │     - Синхронизация по timestamps                                │
  └──────────────────────────────────────────────────────────────────┘
         │
         │ is_outperforming()
         ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  4. Определение зоны                                             │
  │     - Расчет доходностей                                         │
  │     - STANDARDIZED/NORMALIZED/NET_RETURNS                        │
  │     - Green Zone ( outperforming )                               │
  │     - Red Zone ( underperforming )                               │
  └──────────────────────────────────────────────────────────────────┘
         │
         │ Генерация сигнала
         ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  5. Торговый сигнал                                              │
  │     - BUY: cross_up + Green Zone                                 │
  │     - SELL: cross_down + Red Zone                                │
  │     - SL/TP расчет                                               │
  │     - strength (0.0 - 1.0)                                       │
  └──────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────┐
  │    Signal    │
  └──────────────┘
```

---

## Зависимости от других модулей

| Модуль | Использование |
|--------|---------------|
| `kernels.py` | KernelRegression - расчет ядра регрессии и полос |
| `data_manager.py` | DataManager - получение OHLC данных и market cap |
| `config.py` | Конфигурация параметров стратегии |
| `utils.py` | get_session_start_index - определение начала сессии |

---

## Конфигурационные параметры (config.py)

| Параметр | Описание | Использование |
|----------|----------|---------------|
| `KERNEL_TYPE` | Тип ядра регрессии | `self.kernel` |
| `BANDWIDTH` | Ширина окна ядра | `self.kernel` |
| `DEVIATIONS` | Множитель полос | `self.kernel`, расчет stdev |
| `DISPLAY_MODE` | Режим отображения | `self.performance_filter` |
| `BENCHMARK_SYMBOL` | Бенчмарк (BTC/TOTAL) | `get_benchmark_prices()` |
| `SESSION_TYPE` | Тип сессии | `analyze()` - определение start_index |
| `SESSION_HOURS` | Длина сессии | `get_session_start_index()` |
| `SESSION_START_HOUR` | Час начала сессии | `get_session_start_index()` |
| `SESSION_LENGTH` | Длина Rolling окна | `is_outperforming()` |
| `KLINES_LIMIT` | Количество свечей | `get_klines()` |
| `SL_PERCENT` | % стоп-лосса | Расчет SL |
| `TP_PERCENT` | % тейк-профита | Расчет TP |

---

## Примеры использования

### Базовый анализ

```python
from trading_bot.analyzer import MultiKernelAnalyzer
from trading_bot.data_manager import DataManager

# Инициализация
dm = DataManager()
analyzer = MultiKernelAnalyzer(dm)

# Анализ актива
signal = analyzer.analyze("BTCUSDT")

if signal:
    print(f"Сигнал: {signal.action}")
    print(f"Цена: {signal.price}")
    print(f"Зона: {'Green' if signal.is_outperforming else 'Red'}")
    print(f"Сила: {signal.strength}")
```

### Проверка зоны без сигнала

```python
is_green, perf_ratio = analyzer.get_zone_status("ETHUSDT")
print(f"Зона: {'Green' if is_green else 'Red'}")
print(f"Относительная производительность: {perf_ratio * 100:.2f}%")
```

---

## Логирование

Модуль использует стандартный `logging`:

```python
logger = logging.getLogger(__name__)
```

**Уровни логирования**:
- `DEBUG`: Расчеты первого бара, детали зоны, режимы отображения
- `INFO`: Статус зоны, сгенерированные сигналы, параметры сессии
- `WARNING`: Отсутствие данных, ошибки синхронизации

**Примеры логов**:
```
[FIRST_BAR] Asset: open=50000.000000, close=50100.000000, return=0.002000
[ZONE_CHECK] Open=50000.0000 Close=51000.0000 AssetReturn=2.00% ...
[ZONE_STATUS] BTCUSDT: GREEN ZONE
📊 BUY signal for BTCUSDT: price=51000.000000, zone=GREEN, strength=0.85
```
