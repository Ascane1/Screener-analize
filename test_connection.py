"""
Test script for Bybit API connection using pybit library
"""
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading_bot.config import config


def test_connection():
    """Test Bybit API connection"""
    print("=" * 50)
    print("Bybit API Connection Test")
    print("=" * 50)
    
    # Choose endpoint based on configuration
    endpoint = getattr(config, 'BYBIT_ENDPOINT', 'https://api.bybit.com')
    
    # Determine if using demo endpoint
    is_demo = 'demo' in endpoint.lower()
    is_testnet = config.TESTNET and not is_demo
    
    if is_demo:
        print(f"\n[INFO] Using DEMO TRADING: {endpoint}")
    elif is_testnet:
        print(f"\n[INFO] Using TESTNET: {endpoint}")
    else:
        print(f"\n[INFO] Using MAINNET: {endpoint}")
    
    # Initialize session (HTTP class handles both public and authenticated)
    from pybit.unified_trading import HTTP

    # Unauthenticated session for public endpoints
    public_session = HTTP(testnet=is_testnet, demo=is_demo)
    
    # Test 1: Get server time
    print("\n[Test 1] Checking server time...")
    try:
        time_response = public_session.get_server_time()
        # DEBUG: Log full response structure
        print(f"  [DEBUG] Full response: {time_response}")
        print(f"  [DEBUG] Result keys: {time_response['result'].keys() if 'result' in time_response else 'No result'}")
        # Access the correct key (timeSecond or timeNano)
        if 'timeSecond' in time_response['result']:
            print(f"  ✓ Server time: {time_response['result']['timeSecond']}")
        elif 'timeNano' in time_response['result']:
            print(f"  ✓ Server time: {time_response['result']['timeNano']}")
        else:
            print(f"  ✗ Failed: 'time' key not found in result")
            return False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # Test 2: Get market ticker (BTCUSDT)
    print("\n[Test 2] Checking market ticker (BTCUSDT)...")
    try:
        ticker_response = public_session.get_tickers(category="linear", symbol="BTCUSDT")
        print(f"  ✓ Ticker data retrieved")
        print(f"    Last price: {ticker_response['result']['list'][0]['lastPrice']}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # Test 3: Authenticated endpoints (with API keys)
    print("\n[Test 3] Testing authenticated endpoints...")
    print(f"  [DEBUG] API Key: {config.BYBIT_API_KEY[:10]}...")
    print(f"  [DEBUG] TESTNET: {config.TESTNET}")
    
    account_session = HTTP(
        testnet=is_testnet,
        demo=is_demo,
        api_key=config.BYBIT_API_KEY,
        api_secret=config.BYBIT_API_SECRET
    )
    
    try:
        # First, test with a simpler endpoint - get API key info
        print("  [DEBUG] Testing API key permissions...")
        try:
            api_key_info = account_session.get_api_key_info()
            print(f"  [DEBUG] API Key info: {api_key_info}")
        except Exception as api_err:
            print(f"  [DEBUG] API Key info failed: {api_err}")
        
        # Try wallet balance
        balance_response = account_session.get_wallet_balance(accountType="UNIFIED")
        print(f"  ✓ Wallet balance retrieved")
        print(f"  [DEBUG] Full balance response: {balance_response}")
        coin_data = balance_response['result']['list'][0]['coin']
        usdt_coin = next((c for c in coin_data if c['coin'] == "USDT"), None)
        if usdt_coin:
            # Try different possible keys
            balance = usdt_coin.get('availableBalance') or usdt_coin.get('walletBalance') or usdt_coin.get('balance')
            print(f"    Available balance: {balance}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)