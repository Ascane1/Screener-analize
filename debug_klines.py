import sys
sys.path.insert(0, '.')
from trading_bot.data_manager import DataManager

print('Отладка получения klines...')

dm = DataManager()

try:
    response = dm.client.get_kline(
        category="linear",
        symbol="BTCUSDT",
        interval="15",
        limit=5
    )

    print("Raw response:")
    print(response)

    if response and response.get('retCode') == 0:
        data = response['result']['list']
        print(f"\nПолучено {len(data)} свечей")
        for i, kline in enumerate(data[:2]):
            print(f"Свеча {i}: {kline}")
            # Проверим типы данных
            for j, val in enumerate(kline):
                print(f"  [{j}]: {val} (type: {type(val)})")

except Exception as e:
    print(f"Ошибка: {e}")
    import traceback
    traceback.print_exc()