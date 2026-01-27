"""
Diagnostic script to check proxy settings and Bybit API connectivity
"""
import os
import sys

def check_proxy_settings():
    """Check proxy environment variables"""
    print("=" * 60)
    print("PROXY SETTINGS DIAGNOSTIC")
    print("=" * 60)
    
    proxy_vars = [
        'HTTP_PROXY', 'http_proxy',
        'HTTPS_PROXY', 'https_proxy',
        'ALL_PROXY', 'all_proxy',
        'NO_PROXY', 'no_proxy'
    ]
    
    found_proxies = {}
    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            found_proxies[var] = value
            print(f"[FOUND] {var} = {value}")
    
    if not found_proxies:
        print("[OK] No proxy environment variables found")
    else:
        print(f"\n[WARN] Found {len(found_proxies)} proxy variable(s)")
    
    return found_proxies


def check_bybit_connection():
    """Test direct connection to Bybit API"""
    print("\n" + "=" * 60)
    print("BYBIT API CONNECTION TEST")
    print("=" * 60)
    
    import requests
    
    # Test endpoints
    endpoints = [
        ("api-demo.bybit.com", "https://api-demo.bybit.com/v5/market/time"),
        ("api.bybit.com", "https://api.bybit.com/v5/market/time"),
    ]
    
    for name, url in endpoints:
        print(f"\n[Test] Connecting to {name}...")
        try:
            response = requests.get(url, timeout=10)
            print(f"  [OK] Status: {response.status_code}")
            print(f"  [OK] Response: {response.text[:200]}")
        except requests.exceptions.ProxyError as e:
            print(f"  [ERROR] Proxy Error: {e}")
        except requests.exceptions.ConnectionError as e:
            print(f"  [ERROR] Connection Error: {e}")
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")


def check_pybit_connection():
    """Test connection using pybit library"""
    print("\n" + "=" * 60)
    print("PYBIT LIBRARY CONNECTION TEST")
    print("=" * 60)
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from trading_bot.config import config
    
    from pybit.unified_trading import HTTP
    
    endpoint = config.BYBIT_ENDPOINT
    is_demo = 'demo' in endpoint.lower()
    is_testnet = config.TESTNET and not is_demo
    
    print(f"[INFO] BYBIT_ENDPOINT: {endpoint}")
    print(f"[INFO] is_demo: {is_demo}")
    print(f"[INFO] is_testnet: {is_testnet}")
    
    print(f"\n[Test] Creating HTTP session...")
    try:
        session = HTTP(testnet=is_testnet, demo=is_demo)
        print(f"  [OK] Session created")
        
        print(f"\n[Test] Getting server time...")
        response = session.get_server_time()
        print(f"  [OK] Response: {response}")
        
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_proxy_settings()
    check_bybit_connection()
    check_pybit_connection()
