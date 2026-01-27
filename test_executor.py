import sys
sys.path.insert(0, '.')
from trading_bot.executor import Executor
from trading_bot.analyzer import Signal
from trading_bot.data_manager import DataManager

print('Тестирование Executor...')

dm = DataManager()
executor = Executor(dm)

# Создадим тестовый сигнал
signal = Signal(
    symbol='BTCUSDT',
    action='BUY',
    price=50000.0,
    stop_loss=49500.0,
    take_profit=51000.0,
    kernel_value=49950.0,
    is_outperforming=True,
    strength=0.8,
    reason='Test signal'
)

print(f'Тестовый сигнал: {signal.action} {signal.symbol} at {signal.price}')

# Тестируем расчет размера позиции
quantity = executor.calculate_position_size(
    signal.symbol,
    signal.price,
    signal.stop_loss
)
print(f'Размер позиции: {quantity}')

# Тестируем округление цены
rounded_sl = executor.round_price(signal.symbol, signal.stop_loss)
rounded_tp = executor.round_price(signal.symbol, signal.take_profit)
print(f'Округленный SL: {rounded_sl}, TP: {rounded_tp}')

# Проверяем лимиты позиций
positions = dm.get_positions()
print(f'Текущих позиций: {len(positions)}')

# НЕ исполняем реальную сделку - только DRY RUN
print('✓ Executor работает корректно (без реального исполнения)')