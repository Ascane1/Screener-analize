"""Debug script to check Decimal serialization"""
import json
from decimal import Decimal

# Test Decimal serialization
qty = Decimal('50895')
qty_float = 50895.0
qty_str = '50895'

print("=== Testing Decimal serialization ===")
print(f"Decimal('50895'): {repr(qty)}")
print(f"str(Decimal('50895')): {str(qty)}")
print()

# Test JSON serialization
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)

data = {
    'qty_decimal': qty,
    'qty_float': qty_float,
    'qty_str': qty_str
}

print("=== JSON serialization ===")
print(f"Default JSON: {json.dumps(data)}")
print(f"Custom DecimalEncoder: {json.dumps(data, cls=DecimalEncoder)}")
print()

# Test what pybit might do
print("=== What pybit might do ===")
# Some libraries multiply by 10^8 for integer representation
qty_int = int(qty * 100000000)
print(f"int(Decimal('50895') * 100000000): {qty_int}")
print()

# Check if the error matches
error_qty = 5089500000000
calculated_qty = int(qty * 100000000)
print(f"=== Comparison ===")
print(f"Error shows: {error_qty}")
print(f"Calculated (qty * 100000000): {calculated_qty}")
print(f"Match: {error_qty == calculated_qty}")
