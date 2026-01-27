import os
import json
import time
import asyncio
import logging
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union
from functools import wraps
from dataclasses import dataclass, asdict
import requests

logger = logging.getLogger(__name__)


# ============================================
# RETRY DECORATOR
# ============================================

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Декоратор для повторных попыток при ошибках
    
    Args:
        max_attempts: Максимальное количество попыток
        delay: Начальная задержка между попытками
        backoff: Множитель задержки
        exceptions: Типы исключений для перехвата
    """
    def decorator(func: Callable):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"{func.__name__} attempt {attempt}/{max_attempts} "
                            f"failed: {e}. Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
            
            raise last_exception
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"{func.__name__} attempt {attempt}/{max_attempts} "
                            f"failed: {e}. Retrying in {current_delay:.1f}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# ============================================
# RATE LIMITER
# ============================================

class RateLimiter:
    """Ограничитель частоты запросов"""
    
    def __init__(self, max_requests: int, period: float):
        """
        Args:
            max_requests: Максимум запросов за период
            period: Период в секундах
        """
        self.max_requests = max_requests
        self.period = period
        self.requests: List[float] = []
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Ожидать разрешения на запрос"""
        async with self._lock:
            now = time.time()
            
            # Удаляем старые запросы
            self.requests = [
                req for req in self.requests 
                if now - req < self.period
            ]
            
            if len(self.requests) >= self.max_requests:
                # Ждём до освобождения слота
                sleep_time = self.period - (now - self.requests[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self.requests.append(time.time())
    
    def acquire_sync(self):
        """Синхронная версия"""
        now = time.time()
        
        self.requests = [
            req for req in self.requests 
            if now - req < self.period
        ]
        
        if len(self.requests) >= self.max_requests:
            sleep_time = self.period - (now - self.requests[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.requests.append(time.time())


# ============================================
# NOTIFICATIONS
# ============================================

class TelegramNotifier:
    """Отправка уведомлений в Telegram"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    @retry(max_attempts=3, delay=1.0)
    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Отправить сообщение"""
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True
                },
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
    
    def send_signal(
        self, 
        action: str, 
        symbol: str, 
        price: float, 
        sl: float, 
        tp: float,
        reason: str = ""
    ):
        """Отправить уведомление о сигнале"""
        emoji = "🟢" if action == "BUY" else "🔴"
        
        message = f"""
{emoji} <b>{action} {symbol}</b>

💰 Entry: <code>{price:.6f}</code>
🛑 SL: <code>{sl:.6f}</code>
🎯 TP: <code>{tp:.6f}</code>

📝 {reason}
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message)
    
    def send_position_update(self, positions: List[dict]):
        """Отправить обновление по позициям"""
        if not positions:
            return
        
        lines = ["📊 <b>Open Positions</b>\n"]
        
        total_pnl = 0
        for pos in positions:
            pnl = pos['unrealized_pnl']
            total_pnl += pnl
            emoji = "📈" if pnl >= 0 else "📉"
            
            lines.append(
                f"{emoji} {pos['symbol']} {pos['side']}: "
                f"<code>{pnl:+.2f}</code> USDT"
            )
        
        lines.append(f"\n💵 Total PnL: <code>{total_pnl:+.2f}</code> USDT")
        
        self.send_message("\n".join(lines))


class DiscordNotifier:
    """Отправка уведомлений в Discord"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    @retry(max_attempts=3, delay=1.0)
    def send_message(self, content: str = None, embed: dict = None) -> bool:
        """Отправить сообщение через webhook"""
        try:
            payload = {}
            if content:
                payload["content"] = content
            if embed:
                payload["embeds"] = [embed]
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            return response.status_code in [200, 204]
        except Exception as e:
            logger.error(f"Discord send error: {e}")
            return False
    
    def send_signal(
        self, 
        action: str, 
        symbol: str, 
        price: float, 
        sl: float, 
        tp: float,
        reason: str = ""
    ):
        """Отправить уведомление о сигнале"""
        color = 0x00ff00 if action == "BUY" else 0xff0000
        
        embed = {
            "title": f"{'🟢' if action == 'BUY' else '🔴'} {action} {symbol}",
            "color": color,
            "fields": [
                {"name": "💰 Entry", "value": f"`{price:.6f}`", "inline": True},
                {"name": "🛑 Stop Loss", "value": f"`{sl:.6f}`", "inline": True},
                {"name": "🎯 Take Profit", "value": f"`{tp:.6f}`", "inline": True},
                {"name": "📝 Reason", "value": reason, "inline": False},
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.send_message(embed=embed)


# ============================================
# FORMATTING
# ============================================

def format_price(price: float, decimals: int = None) -> str:
    """Форматировать цену"""
    if decimals is None:
        if price >= 1000:
            decimals = 2
        elif price >= 1:
            decimals = 4
        elif price >= 0.01:
            decimals = 6
        else:
            decimals = 8
    
    return f"{price:.{decimals}f}"


def format_percent(value: float, decimals: int = 2) -> str:
    """Форматировать процент"""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Форматировать число с разделителями"""
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.{decimals}f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"


def format_duration(seconds: float) -> str:
    """Форматировать длительность"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_timestamp(dt: datetime = None) -> str:
    """Форматировать timestamp"""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ============================================
# CALCULATIONS
# ============================================

def calculate_pnl(
    entry_price: float,
    current_price: float,
    size: float,
    side: str,
    leverage: int = 1
) -> float:
    """Рассчитать PnL"""
    if side.upper() in ["BUY", "LONG"]:
        pnl = (current_price - entry_price) * size
    else:
        pnl = (entry_price - current_price) * size
    
    return pnl


def calculate_pnl_percent(
    entry_price: float,
    current_price: float,
    side: str,
    leverage: int = 1
) -> float:
    """Рассчитать PnL в процентах"""
    if side.upper() in ["BUY", "LONG"]:
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
    else:
        pnl_pct = ((entry_price - current_price) / entry_price) * 100
    
    return pnl_pct * leverage


def calculate_position_size(
    balance: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float,
    leverage: int = 1
) -> float:
    """
    Рассчитать размер позиции на основе риска
    
    Args:
        balance: Баланс аккаунта
        risk_percent: Риск на сделку (0.01 = 1%)
        entry_price: Цена входа
        stop_loss: Цена стоп-лосса
        leverage: Плечо
    
    Returns:
        Размер позиции в базовом активе
    """
    risk_amount = balance * risk_percent
    stop_distance = abs(entry_price - stop_loss)
    
    if stop_distance == 0:
        return 0.0
    
    position_value = risk_amount / (stop_distance / entry_price)
    position_size = position_value / entry_price
    
    return position_size


def calculate_liquidation_price(
    entry_price: float,
    leverage: int,
    side: str,
    maintenance_margin: float = 0.005  # 0.5%
) -> float:
    """Рассчитать цену ликвидации"""
    if side.upper() in ["BUY", "LONG"]:
        liq_price = entry_price * (1 - (1 / leverage) + maintenance_margin)
    else:
        liq_price = entry_price * (1 + (1 / leverage) - maintenance_margin)
    
    return liq_price


def calculate_risk_reward(
    entry_price: float,
    stop_loss: float,
    take_profit: float
) -> float:
    """Рассчитать Risk/Reward ratio"""
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    
    if risk == 0:
        return 0.0
    
    return reward / risk


# ============================================
# DATA PERSISTENCE
# ============================================

@dataclass
class Trade:
    """Структура сделки для хранения"""
    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: Optional[float]
    pnl_percent: Optional[float]
    entry_time: str
    exit_time: Optional[str]
    stop_loss: float
    take_profit: float
    reason: str
    status: str  # 'open', 'closed', 'cancelled'


class TradeLogger:
    """Логирование сделок в файл"""
    
    def __init__(self, filepath: str = "trades.json"):
        self.filepath = filepath
        self.trades: List[Trade] = []
        self._load()
    
    def _load(self):
        """Загрузить сделки из файла"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.trades = [Trade(**t) for t in data]
            except Exception as e:
                logger.error(f"Error loading trades: {e}")
                self.trades = []
    
    def _save(self):
        """Сохранить сделки в файл"""
        try:
            with open(self.filepath, 'w') as f:
                json.dump([asdict(t) for t in self.trades], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trades: {e}")
    
    def add_trade(self, trade: Trade):
        """Добавить сделку"""
        self.trades.append(trade)
        self._save()
    
    def update_trade(self, trade_id: str, **kwargs):
        """Обновить сделку"""
        for trade in self.trades:
            if trade.id == trade_id:
                for key, value in kwargs.items():
                    if hasattr(trade, key):
                        setattr(trade, key, value)
                self._save()
                return True
        return False
    
    def close_trade(
        self, 
        trade_id: str, 
        exit_price: float, 
        pnl: float,
        pnl_percent: float
    ):
        """Закрыть сделку"""
        self.update_trade(
            trade_id,
            exit_price=exit_price,
            pnl=pnl,
            pnl_percent=pnl_percent,
            exit_time=format_timestamp(),
            status='closed'
        )
    
    def get_open_trades(self) -> List[Trade]:
        """Получить открытые сделки"""
        return [t for t in self.trades if t.status == 'open']
    
    def get_statistics(self) -> dict:
        """Получить статистику по сделкам"""
        closed = [t for t in self.trades if t.status == 'closed']
        
        if not closed:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0
            }
        
        winners = [t for t in closed if t.pnl and t.pnl > 0]
        losers = [t for t in closed if t.pnl and t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in closed if t.pnl)
        total_wins = sum(t.pnl for t in winners if t.pnl)
        total_losses = abs(sum(t.pnl for t in losers if t.pnl))
        
        return {
            'total_trades': len(closed),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': len(winners) / len(closed) * 100 if closed else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(closed) if closed else 0,
            'avg_win': total_wins / len(winners) if winners else 0,
            'avg_loss': total_losses / len(losers) if losers else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else 0,
        }
    
    def generate_report(self) -> str:
        """Генерировать текстовый отчёт"""
        stats = self.get_statistics()
        
        report = f"""
╔══════════════════════════════════════╗
║         TRADING STATISTICS           ║
╠══════════════════════════════════════╣
║ Total Trades:     {stats['total_trades']:>17} ║
║ Winning Trades:   {stats['winning_trades']:>17} ║
║ Losing Trades:    {stats['losing_trades']:>17} ║
║ Win Rate:         {stats['win_rate']:>16.1f}% ║
╠══════════════════════════════════════╣
║ Total PnL:        ${stats['total_pnl']:>15.2f} ║
║ Average PnL:      ${stats['avg_pnl']:>15.2f} ║
║ Average Win:      ${stats['avg_win']:>15.2f} ║
║ Average Loss:     ${stats['avg_loss']:>15.2f} ║
║ Profit Factor:    {stats['profit_factor']:>17.2f} ║
╚══════════════════════════════════════╝
"""
        return report


# ============================================
# VALIDATION
# ============================================

def validate_symbol(symbol: str) -> bool:
    """Проверить формат символа"""
    if not symbol:
        return False
    
    # Должен заканчиваться на USDT
    if not symbol.endswith('USDT'):
        return False
    
    # Минимальная длина (например, BTCUSDT = 7)
    if len(symbol) < 7:
        return False
    
    return True


def validate_price(price: float) -> bool:
    """Проверить корректность цены"""
    return price is not None and price > 0


def validate_quantity(quantity: float, min_qty: float = 0) -> bool:
    """Проверить корректность количества"""
    return quantity is not None and quantity > min_qty


def validate_order_params(
    symbol: str,
    side: str,
    price: float,
    quantity: float,
    stop_loss: float = None,
    take_profit: float = None
) -> Tuple[bool, str]:
    """
    Валидация параметров ордера
    
    Returns:
        (is_valid, error_message)
    """
    if not validate_symbol(symbol):
        return False, f"Invalid symbol: {symbol}"
    
    if side.upper() not in ['BUY', 'SELL']:
        return False, f"Invalid side: {side}"
    
    if not validate_price(price):
        return False, f"Invalid price: {price}"
    
    if not validate_quantity(quantity):
        return False, f"Invalid quantity: {quantity}"
    
    if stop_loss is not None:
        if not validate_price(stop_loss):
            return False, f"Invalid stop loss: {stop_loss}"
        
        if side.upper() == 'BUY' and stop_loss >= price:
            return False, f"Stop loss must be below entry for BUY"
        
        if side.upper() == 'SELL' and stop_loss <= price:
            return False, f"Stop loss must be above entry for SELL"
    
    if take_profit is not None:
        if not validate_price(take_profit):
            return False, f"Invalid take profit: {take_profit}"
        
        if side.upper() == 'BUY' and take_profit <= price:
            return False, f"Take profit must be above entry for BUY"
        
        if side.upper() == 'SELL' and take_profit >= price:
            return False, f"Take profit must be below entry for SELL"
    
    return True, ""


# ============================================
# TIME UTILITIES
# ============================================

def get_timeframe_ms(timeframe: str) -> int:
    """Получить длительность таймфрейма в миллисекундах"""
    unit = timeframe[-1].lower()
    value = int(timeframe[:-1]) if len(timeframe) > 1 else int(timeframe)
    
    multipliers = {
        'm': 60 * 1000,
        'h': 60 * 60 * 1000,
        'd': 24 * 60 * 60 * 1000,
        'w': 7 * 24 * 60 * 60 * 1000,
    }
    
    # Для Bybit формат "15" означает 15 минут
    if timeframe.isdigit():
        return int(timeframe) * 60 * 1000
    
    return value * multipliers.get(unit, 60 * 1000)


def get_candle_close_time(timeframe: str) -> datetime:
    """Получить время закрытия текущей свечи"""
    now = datetime.now()
    tf_ms = get_timeframe_ms(timeframe)
    tf_seconds = tf_ms // 1000
    
    # Округляем до ближайшего периода
    timestamp = now.timestamp()
    current_period = int(timestamp // tf_seconds) * tf_seconds
    next_close = current_period + tf_seconds
    
    return datetime.fromtimestamp(next_close)


def time_until_candle_close(timeframe: str) -> float:
    """Время до закрытия текущей свечи в секундах"""
    close_time = get_candle_close_time(timeframe)
    now = datetime.now()
    
    delta = close_time - now
    return max(delta.total_seconds(), 0)


def is_new_candle(timeframe: str, last_check: datetime) -> bool:
    """Проверить, появилась ли новая свеча с последней проверки"""
    tf_ms = get_timeframe_ms(timeframe)
    tf_seconds = tf_ms // 1000
    
    last_period = int(last_check.timestamp() // tf_seconds)
    current_period = int(datetime.now().timestamp() // tf_seconds)
    
    return current_period > last_period


# ============================================
# SIGNAL GENERATION HELPERS
# ============================================

def generate_trade_id() -> str:
    """Генерировать уникальный ID сделки"""
    import uuid
    return str(uuid.uuid4())[:8]


def crossover(series1: List[float], series2: List[float]) -> bool:
    """Проверить crossover (series1 пересекает series2 снизу вверх)"""
    if len(series1) < 2 or len(series2) < 2:
        return False
    
    return series1[-2] <= series2[-2] and series1[-1] > series2[-1]


def crossunder(series1: List[float], series2: List[float]) -> bool:
    """Проверить crossunder (series1 пересекает series2 сверху вниз)"""
    if len(series1) < 2 or len(series2) < 2:
        return False
    
    return series1[-2] >= series2[-2] and series1[-1] < series2[-1]


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    # Тест форматирования
    print(format_price(0.00001234))  # 0.00001234
    print(format_price(123.456))      # 123.4560
    print(format_percent(15.5))       # +15.50%
    print(format_number(1234567))     # 1.23M
    
    # Тест расчётов
    print(calculate_pnl(100, 110, 1, "BUY"))  # 10.0
    print(calculate_risk_reward(100, 95, 115))  # 3.0
    
    # Тест времени
    print(time_until_candle_close("15"))  # секунды до закрытия
    print(get_candle_close_time("15"))    # время закрытия