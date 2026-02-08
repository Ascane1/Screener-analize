import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env files
load_dotenv('.env.api')
load_dotenv('.env.kernel')
load_dotenv('.env.performance')
load_dotenv('.env.trading')
load_dotenv('.env.screener')
load_dotenv('.env.system')
load_dotenv('.env.exclusions')


@dataclass
class Config:
    # ============ API ============
    BYBIT_API_KEY: str = field(default_factory=lambda: os.environ.get('BYBIT_API_KEY', 'qlWETV2AQltSG5ca1L'))
    BYBIT_API_SECRET: str = field(default_factory=lambda: os.environ.get('BYBIT_API_SECRET', 'U2wZCBGtFA1c78Kd6t79nwpwqbAZLDhMVb0T'))
    BYBIT_ENDPOINT: str = field(default_factory=lambda: os.environ.get('BYBIT_ENDPOINT', 'https://api-testnet.bybit.com'))
    TESTNET: bool = field(default_factory=lambda: os.environ.get('TESTNET', 'True').lower() == 'true')

    # ============ KERNEL REGRESSION ============
    KERNEL_TYPE: str = field(default_factory=lambda: os.environ.get('KERNEL_TYPE', 'Epanechnikov'))  # Тип ядра
    BANDWIDTH: int = field(default_factory=lambda: int(os.environ.get('BANDWIDTH', '20')))           # Период сглаживания
    SOURCE: str = field(default_factory=lambda: os.environ.get('SOURCE', 'close'))         # Источник данных
    DEVIATIONS: float = field(default_factory=lambda: float(os.environ.get('DEVIATIONS', '2.0')))    # Множитель для полос отклонения (Deviation Bands)

    # ============ RELATIVE PERFORMANCE ============
    BENCHMARK_SYMBOL: str = field(default_factory=lambda: os.environ.get('BENCHMARK_SYMBOL', 'BTCUSDT'))  # Бенчмарк (BTC для крипты)
    DISPLAY_MODE: str = field(default_factory=lambda: os.environ.get('DISPLAY_MODE', 'Standardized'))  # Net Returns, Rescaled Returns, Standardized Returns
    SESSION_LENGTH: int = field(default_factory=lambda: int(os.environ.get('SESSION_LENGTH', '48')))  # Длина сессии в барах (12 часов на 15м = 48 баров)
    SESSION_TYPE: str = field(default_factory=lambda: os.environ.get('SESSION_TYPE', 'Fixed'))  # Тип сессии: 'Fixed' или 'Rolling'
    SESSION_START_HOUR: int = field(default_factory=lambda: int(os.environ.get('SESSION_START_HOUR', '0')))  # Час начала фиксированной сессии (UTC)
    SESSION_HOURS: int = field(default_factory=lambda: int(os.environ.get('SESSION_HOURS', '12')))  # Интервал фиксированной сессии в часах

    # ============ ТОРГОВЛЯ ============
    TIMEFRAME: str = field(default_factory=lambda: os.environ.get('TIMEFRAME', '15'))  # Таймфрейм (минуты для Bybit: "1", "5", "15", "60", etc)
    SL_PERCENT: float = field(default_factory=lambda: float(os.environ.get('SL_PERCENT', '5.0')))   # Stop Loss %
    TP_PERCENT: float = field(default_factory=lambda: float(os.environ.get('TP_PERCENT', '5.0')))   # Take Profit %
    LEVERAGE: int = field(default_factory=lambda: int(os.environ.get('LEVERAGE', '15')))
    RISK_PER_TRADE: float = field(default_factory=lambda: float(os.environ.get('RISK_PER_TRADE', '0.01')))  # 1% от депозита
    MAX_POSITIONS: int = field(default_factory=lambda: int(os.environ.get('MAX_POSITIONS', '5')))

    # ============ СКРИНЕР ============
    TOP_GAINERS_COUNT: int = field(default_factory=lambda: int(os.environ.get('TOP_GAINERS_COUNT', '5')))
    TOP_LOSERS_COUNT: int = field(default_factory=lambda: int(os.environ.get('TOP_LOSERS_COUNT', '5')))
    MIN_VOLUME_24H: float = field(default_factory=lambda: float(os.environ.get('MIN_VOLUME_24H', '1000000')))
    MIN_PRICE_CHANGE: float = field(default_factory=lambda: float(os.environ.get('MIN_PRICE_CHANGE', '2.0')))

    # ============ СИСТЕМА ============
    SCAN_INTERVAL_SECONDS: int = field(default_factory=lambda: int(os.environ.get('SCAN_INTERVAL_SECONDS', '60')))
    KLINES_LIMIT: int = field(default_factory=lambda: int(os.environ.get('KLINES_LIMIT', '200')))

    # ============ ИСКЛЮЧЕНИЯ ============
    EXCLUDED_SYMBOLS: List[str] = field(default_factory=lambda: os.environ.get('EXCLUDED_SYMBOLS', 'USDCUSDT,BTCUSDT').split(','))


config = Config()