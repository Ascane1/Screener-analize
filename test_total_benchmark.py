"""
Тестирование корректности работы TOTAL бенчмарка с историческими данными.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from trading_bot.data_manager import DataManager
from trading_bot.analyzer import MultiKernelAnalyzer, DisplayMode, RelativePerformanceFilter
from trading_bot.config import config
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def test_total_market_cap_history():
    """Тест получения исторических данных TOTAL market cap"""
    print("\n" + "="*60)
    print("ТЕСТ 1: Получение исторических данных TOTAL market cap")
    print("="*60)
    
    dm = DataManager()
    df = dm.get_total_market_cap_history(days=7)
    
    if df is None or df.empty:
        print("❌ ОШИБКА: Не удалось получить исторические данные")
        return False
    
    print(f"✅ Получено {len(df)} точек данных")
    print(f"   Диапазон: {df['timestamp'].min()} - {df['timestamp'].max()}")
    print(f"   Market Cap min: ${df['market_cap'].min():,.0f}")
    print(f"   Market Cap max: ${df['market_cap'].max():,.0f}")
    print(f"   Изменение: {((df['market_cap'].iloc[-1] / df['market_cap'].iloc[0]) - 1) * 100:+.2f}%")
    
    # Проверяем что данные изменяются (не все одинаковые)
    if df['market_cap'].nunique() < 10:
        print("❌ ПРЕДУПРЕЖДЕНИЕ: Слишком мало уникальных значений!")
        return False
    
    return True


def test_total_benchmark_alignment():
    """Тест выравнивания TOTAL бенчмарка по таймстемпам актива"""
    print("\n" + "="*60)
    print("ТЕСТ 2: Выравнивание TOTAL бенчмарка по таймстемпам")
    print("="*60)
    
    dm = DataManager()
    analyzer = MultiKernelAnalyzer(dm)
    
    # Получаем данные для тестового символа
    symbol = "BTCUSDT"
    df = dm.get_klines(symbol, limit=200)
    
    if df.empty:
        print(f"❌ ОШИБКА: Не удалось получить данные для {symbol}")
        return False
    
    print(f"✅ Получены данные для {symbol}: {len(df)} баров")
    print(f"   Время первого бара: {df['timestamp'].iloc[0]}")
    print(f"   Время последнего бара: {df['timestamp'].iloc[-1]}")
    
    # Получаем бенчмарк с выравниванием
    asset_timestamps = df['timestamp'].values
    benchmark_prices = analyzer._get_total_benchmark(len(df), asset_timestamps)
    
    if len(benchmark_prices) == 0:
        print("❌ ОШИБКА: Не удалось получить бенчмарк")
        return False
    
    print(f"✅ Получен бенчмарк: {len(benchmark_prices)} точек")
    print(f"   Диапазон: ${benchmark_prices.min():,.0f} - ${benchmark_prices.max():,.0f}")
    print(f"   Изменение: {((benchmark_prices[-1] / benchmark_prices[0]) - 1) * 100:+.2f}%")
    
    # Проверяем что длины совпадают
    if len(benchmark_prices) != len(df):
        print(f"❌ ОШИБКА: Длина бенчмарка ({len(benchmark_prices)}) != длина актива ({len(df)})")
        return False
    
    # Проверяем что данные изменяются (не все одинаковые)
    if np.std(benchmark_prices) < 1e-6:
        print("❌ ОШИБКА: Бенчмарк не изменяется!")
        return False
    
    print("✅ Длины совпадают, бенчмарк изменяется")
    return True


def test_zone_calculation():
    """Тест расчёта зон с TOTAL бенчмарком"""
    print("\n" + "="*60)
    print("ТЕСТ 3: Расчёт зон с TOTAL бенчмарком")
    print("="*60)
    
    dm = DataManager()
    analyzer = MultiKernelAnalyzer(dm)
    
    # Тестовые символы
    symbols = ["BTCUSDT", "ETHUSDT"]
    
    for symbol in symbols:
        print(f"\n--- Тестируем {symbol} ---")
        
        df = dm.get_klines(symbol, limit=200)
        if df.empty:
            print(f"❌ Пропуск {symbol} - нет данных")
            continue
        
        # Рассчитываем hlc3
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        asset_prices = df['hlc3'].values
        
        # Получаем бенчмарк
        asset_timestamps = df['timestamp'].values
        benchmark_prices = analyzer._get_total_benchmark(len(df), asset_timestamps)
        
        if len(benchmark_prices) == 0:
            print(f"❌ ОШИБКА: Нет бенчмарка для {symbol}")
            continue
        
        # Рассчитываем зону
        is_green = analyzer.performance_filter.is_outperforming(
            asset_prices,
            benchmark_prices,
            session_length=config.SESSION_LENGTH
        )
        
        # Рассчитываем производительность
        perf_ratio = analyzer.performance_filter.get_performance_ratio(
            asset_prices, benchmark_prices
        )
        
        zone = "🟢 GREEN" if is_green else "🔴 RED"
        print(f"   {symbol}: {zone} | Perf: {perf_ratio*100:+.2f}%")
        print(f"   Asset return: {((asset_prices[-1]/asset_prices[0])-1)*100:+.2f}%")
        print(f"   Benchmark return: {((benchmark_prices[-1]/benchmark_prices[0])-1)*100:+.2f}%")
    
    return True


def test_comparison_with_fixed_btc():
    """Сравнение зон TOTAL vs BTCUSDT бенчмарка"""
    print("\n" + "="*60)
    print("ТЕСТ 4: Сравнение TOTAL vs BTCUSDT бенчмарка")
    print("="*60)
    
    dm = DataManager()
    analyzer = MultiKernelAnalyzer(dm)
    
    symbol = "ETHUSDT"
    df = dm.get_klines(symbol, limit=200)
    
    if df.empty:
        print("❌ Нет данных для теста")
        return False
    
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    asset_prices = df['hlc3'].values
    asset_timestamps = df['timestamp'].values
    
    # TOTAL бенчмарк
    total_prices = analyzer._get_total_benchmark(len(df), asset_timestamps)
    is_green_total = analyzer.performance_filter.is_outperforming(
        asset_prices, total_prices, session_length=config.SESSION_LENGTH
    )
    
    # BTCUSDT бенчмарк
    df_btc = dm.get_klines("BTCUSDT", limit=250)
    if not df_btc.empty:
        df_btc['hlc3'] = (df_btc['high'] + df_btc['low'] + df_btc['close']) / 3
        
        # Синхронизация по таймстемпам
        asset_ts_series = pd.Series(asset_timestamps, name='timestamp')
        merged = pd.merge(
            asset_ts_series,
            df_btc[['timestamp', 'hlc3']],
            on='timestamp',
            how='inner'
        )
        
        if len(merged) > 0:
            btc_prices = merged['hlc3'].values
            is_green_btc = analyzer.performance_filter.is_outperforming(
                asset_prices[:len(btc_prices)], btc_prices, session_length=config.SESSION_LENGTH
            )
            
            print(f"   {symbol}:")
            print(f"   TOTAL бенчмарк: {'🟢 GREEN' if is_green_total else '🔴 RED'}")
            print(f"   BTCUSDT бенчмарк: {'🟢 GREEN' if is_green_btc else '🔴 RED'}")
            
            if is_green_total != is_green_btc:
                print("   ⚠️ Разница в определении зоны между TOTAL и BTC!")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("НАЧАЛО ТЕСТИРОВАНИЯ TOTAL БЕНЧМАРКА")
    print("="*60)
    
    results = []
    
    try:
        results.append(("Исторические данные", test_total_market_cap_history()))
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        results.append(("Исторические данные", False))
    
    try:
        results.append(("Выравнивание", test_total_benchmark_alignment()))
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        results.append(("Выравнивание", False))
    
    try:
        results.append(("Расчёт зон", test_zone_calculation()))
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        results.append(("Расчёт зон", False))
    
    try:
        results.append(("Сравнение с BTC", test_comparison_with_fixed_btc()))
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        results.append(("Сравнение с BTC", False))
    
    # Итоги
    print("\n" + "="*60)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("="*60)
    
    for name, passed in results:
        status = "✅ ПРОЙДЕН" if passed else "❌ НЕ ПРОЙДЕН"
        print(f"   {name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + ("🎉 Все тесты пройдены!" if all_passed else "⚠️ Есть ошибки, требуется доработка"))
