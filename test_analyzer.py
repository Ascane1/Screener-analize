import sys
sys.path.insert(0, '.')
from trading_bot.analyzer import MultiKernelAnalyzer
from trading_bot.data_manager import DataManager

print('Тестирование Analyzer...')

dm = DataManager()
analyzer = MultiKernelAnalyzer(dm)

# Тестируем на одном символе
symbol = 'BTCUSDT'
print(f'Анализ {symbol}...')

signal = analyzer.analyze(symbol)

if signal:
    print(f'✓ Сигнал найден: {signal.action} {signal.symbol}')
    print(f'  Цена: {signal.price:.6f}')
    print(f'  Зона: {"GREEN" if signal.is_outperforming else "RED"}')
    print(f'  Сила: {signal.strength:.2f}')
    print(f'  Причина: {signal.reason}')
else:
    print(f'✗ Сигнал не найден для {symbol}')

# Тестируем статус зоны
is_green, perf_ratio = analyzer.get_zone_status(symbol)
print(f'Статус зоны {symbol}: {"GREEN" if is_green else "RED"} (perf: {perf_ratio:.4f})')

print('✓ Analyzer работает корректно')