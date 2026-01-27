"""
Debug script for RIVERUSDT symbol
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading_bot.config import config
from trading_bot.data_manager import DataManager

print("=" * 60)
print("RIVERUSDT DEBUG")
print("=" * 60)

print(f"\n[INFO] BYBIT_ENDPOINT: {config.BYBIT_ENDPOINT}")
print(f"[INFO] TESTNET: {config.TESTNET}")

dm = DataManager()

# Test 1: Get all tickers and check if RIVERUSDT exists
print("\n[Test 1] Checking if RIVERUSDT exists in tickers...")
try:
    tickers = dm.get_all_tickers()
    print(f"  [INFO] Total tickers: {len(tickers)}")
    
    if 'RIVERUSDT' in tickers:
        print(f"  [OK] RIVERUSDT found: {tickers['RIVERUSDT']}")
    else:
        print(f"  [WARN] RIVERUSDT NOT found in tickers list")
        # Show similar symbols
        similar = [s for s in tickers.keys() if 'RIVER' in s.upper()]
        print(f"  [INFO] Similar symbols: {similar}")
except Exception as e:
    print(f"  [ERROR] {type(e).__name__}: {e}")

# Test 2: Try to get klines for RIVERUSDT
print("\n[Test 2] Getting klines for RIVERUSDT...")
try:
    klines = dm.get_klines('RIVERUSDT', limit=10)
    if not klines.empty:
        print(f"  [OK] Received {len(klines)} klines")
        print(f"  [INFO] First kline: {klines.iloc[0]}")
        print(f"  [INFO] Last kline: {klines.iloc[-1]}")
    else:
        print(f"  [WARN] Empty DataFrame returned")
except Exception as e:
    print(f"  [ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Try another symbol to compare
print("\n[Test 3] Getting klines for BTCUSDT (comparison)...")
try:
    btc_klines = dm.get_klines('BTCUSDT', limit=5)
    if not btc_klines.empty:
        print(f"  [OK] Received {len(btc_klines)} klines for BTCUSDT")
    else:
        print(f"  [WARN] Empty DataFrame returned for BTCUSDT")
except Exception as e:
    print(f"  [ERROR] {type(e).__name__}: {e}")
