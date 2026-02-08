"""Debug script to check qty calculation for PONKEUSDT"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decimal import Decimal, ROUND_DOWN
from trading_bot.data_manager import DataManager

dm = DataManager()

# Get instrument info for PONKEUSDT
response = dm.client.get_instruments_info(
    category="linear",
    symbol="PONKEUSDT"
)

if response and response.get('retCode') == 0:
    info = response['result']['list'][0]
    lot_size = info['lotSizeFilter']
    
    print("=== PONKEUSDT Instrument Info ===")
    print(f"lotSizeFilter: {lot_size}")
    print()
    
    min_order_qty = float(lot_size['minOrderQty'])
    qty_step = float(lot_size['qtyStep'])
    max_order_qty = float(lot_size.get('maxOrderQty', 'N/A'))
    
    print(f"minOrderQty: {min_order_qty}")
    print(f"qtyStep: {qty_step}")
    print(f"maxOrderQty: {max_order_qty}")
    print()
    
    # Test the round_qty logic
    quantity = 50895.0
    
    print(f"=== Testing round_qty logic ===")
    print(f"Input quantity: {quantity}")
    print()
    
    # Using Decimal
    qty_step_decimal = Decimal(str(qty_step))
    qty_decimal = Decimal(str(quantity))
    
    print(f"qty_step as Decimal: {qty_step_decimal}")
    print(f"quantity as Decimal: {qty_decimal}")
    print()
    
    steps_count = (qty_decimal / qty_step_decimal).to_integral_value(rounding=ROUND_DOWN)
    print(f"steps_count: {steps_count}")
    
    rounded_qty = steps_count * qty_step_decimal
    print(f"rounded_qty (before formatting): {rounded_qty}")
    print(f"rounded_qty (as float): {float(rounded_qty)}")
    print()
    
    # Determine precision
    step_str = format(qty_step, 'f').rstrip('0')
    precision = len(step_str.split('.')[1]) if '.' in step_str else 0
    print(f"step_str: {step_str}")
    print(f"precision: {precision}")
    print()
    
    # Format result
    result = f"{rounded_qty:.{precision}f}"
    print(f"Formatted result: '{result}'")
    print(f"Result type: {type(result)}")
    print()
    
    # Check if this matches the error
    error_qty = 5089500000000
    print(f"=== Comparison with error ===")
    print(f"Error shows order_qty: {error_qty}")
    print(f"Our result: {result}")
    
    # Try to convert result to see what API might receive
    try:
        result_int = int(float(result) * 100000000)
        print(f"Result * 100000000: {result_int}")
    except Exception as e:
        print(f"Error converting: {e}")
    
else:
    print(f"Error fetching instrument info: {response}")
