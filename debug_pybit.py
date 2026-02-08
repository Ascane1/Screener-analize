"""Debug script to check pybit behavior with qty"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading_bot.data_manager import DataManager
import json

dm = DataManager()

# Test what happens when we call place_order with different qty types
print("=== Testing place_order with different qty types ===")

# First, let's check the request that would be sent
# We'll use a mock to see the actual request

# Let's check the source code of pybit to understand how it handles params
import pybit.unified_trading
import inspect

print("\n=== Checking pybit.unified_trading.HTTP.place_order signature ===")
if hasattr(pybit.unified_trading.HTTP, 'place_order'):
    sig = inspect.signature(pybit.unified_trading.HTTP.place_order)
    print(f"Signature: {sig}")
else:
    print("place_order not found in HTTP class")

print("\n=== Checking _request method ===")
if hasattr(pybit.unified_trading.HTTP, '_request'):
    print("_request method found")
else:
    print("_request not found")

# Let's check how pybit handles JSON serialization
print("\n=== Checking pybit JSON handling ===")
try:
    from pybit import utils
    print(f"utils module: {utils}")
    if hasattr(utils, 'json'):
        print(f"utils.json: {utils.json}")
except Exception as e:
    print(f"Error checking utils: {e}")

# Check if there's a custom JSON encoder
print("\n=== Checking for custom JSON encoder in pybit ===")
import pybit
print(f"pybit version: {pybit.__version__}")

# Let's look at the actual request method
print("\n=== Looking at _request method source ===")
source = inspect.getsource(pybit.unified_trading.HTTP._request)
print("First 100 lines of _request method:")
print('\n'.join(source.split('\n')[:100]))
