import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if 'trading_bot' in os.path.dirname(os.path.abspath(__file__)):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pybit.unified_trading import HTTP
from trading_bot.config import config
import logging
import time
import requests
from datetime import datetime

logger = logging.getLogger(__name__)


class DataManager:
    """Управление данными с биржи"""
    
    def __init__(self):
        # Determine if using demo endpoint
        is_demo = 'demo' in config.BYBIT_ENDPOINT.lower()
        is_testnet = config.TESTNET and not is_demo
        
        self.client = HTTP(
            testnet=is_testnet,
            demo=is_demo,
            api_key=config.BYBIT_API_KEY,
            api_secret=config.BYBIT_API_SECRET,
        )
        self._klines_cache: Dict[str, pd.DataFrame] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = 30  # Секунд
        self._global_data_cache = None
        self._global_cache_timestamp = 0
        self._global_cache_ttl = 300  # 5 минут для CoinGecko
    
    def get_all_tickers(self) -> Dict[str, dict]:
        """Получить все тикеры USDT Perpetual"""
        try:
            response = self.client.get_tickers(category="linear")
            tickers = {}
            
            for item in response['result']['list']:
                symbol = item['symbol']
                if symbol.endswith('USDT') and 'USDC' not in symbol:
                    tickers[symbol] = {
                        'symbol': symbol,
                        'last_price': float(item['lastPrice']),
                        'price_change_24h': float(item['price24hPcnt']) * 100,
                        'volume_24h': float(item['turnover24h']),
                        'high_24h': float(item['highPrice24h']),
                        'low_24h': float(item['lowPrice24h']),
                    }
            
            return tickers
        except Exception as e:
            logger.error(f"Error fetching tickers: {e}")
            return {}
    
    def get_klines(self, symbol: str, limit: int = 200) -> pd.DataFrame:
        """Получить свечи с кешированием"""
        cache_key = f"{symbol}_{config.TIMEFRAME}_{limit}"
        current_time = time.time()
        
        # Проверяем кеш
        if cache_key in self._klines_cache:
            if current_time - self._cache_timestamps.get(cache_key, 0) < self._cache_ttl:
                return self._klines_cache[cache_key].copy()
        
        try:
            response = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=config.TIMEFRAME,
                limit=limit
            )
            
            data = response['result']['list']
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Кешируем
            self._klines_cache[cache_key] = df
            self._cache_timestamps[cache_key] = current_time
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_account_balance(self) -> float:
        """Получить баланс USDT"""
        try:
            response = self.client.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT"
            )
            coin_data = response['result']['list'][0]['coin']
            usdt_coin = next((c for c in coin_data if c['coin'] == "USDT"), None)
            if usdt_coin:
                # Try different possible keys
                balance = float(usdt_coin.get('availableBalance') or usdt_coin.get('walletBalance') or usdt_coin.get('balance') or 0)
                return balance
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0
    
    def get_positions(self) -> List[dict]:
        """Получить открытые позиции"""
        try:
            response = self.client.get_positions(
                category="linear",
                settleCoin="USDT"
            )
            positions = []
            for pos in response['result']['list']:
                if float(pos['size']) > 0:
                    positions.append({
                        'symbol': pos['symbol'],
                        'side': pos['side'],
                        'size': float(pos['size']),
                        'entry_price': float(pos['avgPrice']),
                        'unrealized_pnl': float(pos['unrealisedPnl']),
                        'leverage': pos['leverage']
                    })
            return positions
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    def get_global_market_data(self) -> Optional[Dict]:
        """
        Получает глобальные данные рынка через CoinGecko API /global.
        Включает total_market_cap, total_volume, market_cap_percentage и др.
        """
        current_time = time.time()

        # Проверка кеша (5 минут)
        if self._global_data_cache and (current_time - self._global_cache_timestamp < self._global_cache_ttl):
            return self._global_data_cache

        try:
            url = "https://api.coingecko.com/api/v3/global"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json().get('data', {})

            if data:
                self._global_data_cache = data
                self._global_cache_timestamp = current_time
                logger.info("Global market data updated from CoinGecko")
                return data

        except Exception as e:
            logger.error(f"Error fetching CoinGecko global data: {e}")
            # Если есть старый кеш, возвращаем его при ошибке
            return self._global_data_cache

        return None

    def get_total_market_data(self) -> Optional[Dict]:
        """
        Получить данные общей капитализации крипторынка (TOTAL) через CoinGecko API.
        Используется как бенчмарк для сравнения с отдельными монетами.
        
        Returns:
            Dict с полями 'total_market_cap', 'total_volume', 'market_cap_percentage'
        """
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 1,
                "page": 1,
                "sparkline": False
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data:
                coin = data[0]
                return {
                    'total_market_cap': coin.get('market_cap', 0),
                    'total_volume': coin.get('total_volume', 0),
                    'name': coin.get('name', 'Crypto Market')
                }
                
        except Exception as e:
            logger.error(f"Error fetching TOTAL market data: {e}")
        
        return None

    def get_kline_data(self, symbol: str, interval: str, start_time: int = None, end_time: int = None, limit: int = 200) -> list:
        """
        Получает свечные данные (Kline/OHLCV) для заданного символа и интервала.

        Args:
            symbol (str): Торговый символ (например, "BTCUSDT").
            interval (str): Интервал свечи (например, "1", "60", "D").
                            См. документацию Bybit для полных значений.
            start_time (int, optional): Время начала в миллисекундах. Если не указано, берется последние N свечей.
            end_time (int, optional): Время окончания в миллисекундах.
            limit (int, optional): Количество свечей для получения (макс. 1000).

        Returns:
            list: Список свечных данных. Каждый элемент - это список:
                  [timestamp, open, high, low, close, volume]
        """
        print(f"[DataManager] Получение свечных данных для {symbol}, интервал {interval}...")
        try:
            params = {
                "category": "linear",  # Или "spot", "inverse", в зависимости от потребностей
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            if start_time:
                params["start"] = start_time
            if end_time:
                params["end"] = end_time

            response = self.client.get_kline(**params)

            if response and response.get('retCode') == 0:
                klines = response['result']['list']
                print(f"  Успешно получено {len(klines)} свечей для {symbol}.")
                # Свечи приходят в порядке убывания времени (от новых к старым),
                # для анализа может потребоваться их перевернуть.
                return [list(map(lambda x, idx: int(x) if idx == 0 else float(x), kline, range(len(kline)))) for kline in klines][::-1]
            else:
                print(f"  Ошибка при получении свечей: {response.get('retMsg', 'Неизвестная ошибка')}")
                return []
        except Exception as e:
            print(f"  Исключение при получении свечей: {e}")
            return []

    def clear_cache(self):
        """Очистить кеш"""
        self._klines_cache.clear()
        self._cache_timestamps.clear()
        self._global_data_cache = None
        self._global_cache_timestamp = 0


# Пример использования (можно добавить в trading_bot/main.py или отдельный тестовый файл)
if __name__ == "__main__":
    # Установите config.TESTNET = True для тестовой сети
    # config.TESTNET = True

    dm = DataManager()

    print("\n--- Тест: Получение последних 1-часовых свечей для BTCUSDT ---")
    btc_klines = dm.get_kline_data(symbol="BTCUSDT", interval="60", limit=5)
    if btc_klines:
        print(f"Последняя свеча BTCUSDT (1h): {datetime.fromtimestamp(btc_klines[-1][0]/1000)} - O:{btc_klines[-1][1]}, H:{btc_klines[-1][2]}, L:{btc_klines[-1][3]}, C:{btc_klines[-1][4]}, V:{btc_klines[-1][5]}")
    else:
        print("Не удалось получить свечи BTCUSDT.")

    print("\n--- Тест: Получение последних 15-минутных свечей для ETHUSDT ---")
    eth_klines = dm.get_kline_data(symbol="ETHUSDT", interval="15", limit=3)
    if eth_klines:
        print(f"Последняя свеча ETHUSDT (15m): {datetime.fromtimestamp(eth_klines[-1][0]/1000)} - O:{eth_klines[-1][1]}, H:{eth_klines[-1][2]}, L:{eth_klines[-1][3]}, C:{eth_klines[-1][4]}, V:{eth_klines[-1][5]}")
    else:
        print("Не удалось получить свечи ETHUSDT.")