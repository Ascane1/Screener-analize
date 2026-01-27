import sys
sys.path.insert(0, '.')
from trading_bot.screener import Screener
from trading_bot.data_manager import DataManager

print('Тестирование Screener...')

dm = DataManager()
screener = Screener(dm)

# Получить топ монет
gainers, losers = screener.get_top_movers()

print(f'Топ растущих: {len(gainers)}')
for g in gainers[:3]:
    print(f'  {g["symbol"]}: +{g["price_change_24h"]:.2f}%')

print(f'Топ падающих: {len(losers)}')
for l in losers[:3]:
    print(f'  {l["symbol"]}: {l["price_change_24h"]:.2f}%')

# Получить символы для анализа
symbols = screener.get_symbols_to_analyze()
print(f'Символы для анализа: {symbols}')

print('✓ Screener работает корректно')