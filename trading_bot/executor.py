import math
from decimal import Decimal, ROUND_DOWN
from typing import Optional
from trading_bot.data_manager import DataManager
from trading_bot.analyzer import Signal
from trading_bot.config import config
import logging

logger = logging.getLogger(__name__)


class Executor:
    """Исполнение сделок"""
    
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager
    
    def _get_instrument_info(self, symbol: str) -> Optional[dict]:
        """Получить информацию об инструменте"""
        try:
            response = self.dm.client.get_instruments_info(
                category="linear",
                symbol=symbol
            )
            info = response['result']['list'][0]
            return {
                'min_qty': float(info['lotSizeFilter']['minOrderQty']),
                'max_qty': float(info['lotSizeFilter'].get('maxOrderQty', float('inf'))),
                'qty_step': float(info['lotSizeFilter']['qtyStep']),
                'tick_size': float(info['priceFilter']['tickSize']),
                'min_notional': float(info['lotSizeFilter'].get('minNotionalValue', 0))
            }
        except Exception as e:
            logger.error(f"Error getting instrument info: {e}")
            return None
    
    def calculate_position_size(
        self, 
        symbol: str, 
        entry_price: float, 
        stop_loss: float
    ) -> float:
        """Расчёт размера позиции на основе риска"""
        balance = self.dm.get_account_balance()
        risk_amount = balance * config.RISK_PER_TRADE
        
        stop_distance = abs(entry_price - stop_loss) / entry_price
        if stop_distance == 0:
            stop_distance = config.SL_PERCENT / 100
        
        position_value = risk_amount / stop_distance
        quantity = position_value / entry_price
        
        # Округляем до допустимого шага
        info = self._get_instrument_info(symbol)
        if info:
            qty_step = info['qty_step']
            min_qty = info['min_qty']
            max_qty = info['max_qty']
            quantity = math.floor(quantity / qty_step) * qty_step
            quantity = max(quantity, min_qty)
            
            # Проверяем максимальный лимит
            if quantity > max_qty:
                logger.warning(f"Calculated quantity {quantity} exceeds max_qty {max_qty}. Capping to max.")
                quantity = max_qty
        
        logger.info(f"Position size: {quantity} (risk: ${risk_amount:.2f})")
        
        return quantity
    
    def set_leverage(self, symbol: str):
        """Установить плечо"""
        try:
            self.dm.client.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(config.LEVERAGE),
                sellLeverage=str(config.LEVERAGE)
            )
        except Exception as e:
            logger.debug(f"Leverage note: {e}")
    
    def round_price(self, symbol: str, price: float) -> str:
        """Округлить цену до tick size и вернуть как строку"""
        info = self._get_instrument_info(symbol)
        if info:
            tick = Decimal(str(info['tick_size']))
            price_decimal = Decimal(str(price))
            
            # Округляем до nearest tick
            # Используем quantize для точного округления
            ticks_count = (price_decimal / tick).to_integral_value(rounding=ROUND_DOWN)
            rounded_price = ticks_count * tick
            
            # Определяем точность
            tick_str = format(info['tick_size'], 'f').rstrip('0')
            precision = len(tick_str.split('.')[1]) if '.' in tick_str else 0
            
            # Форматируем, убирая trailing zeros
            return f"{rounded_price:.{precision}f}"
        return str(price)
    
    def round_qty(self, symbol: str, quantity: float) -> str:
        """Округлить количество до qty_step и вернуть как строку"""
        info = self._get_instrument_info(symbol)
        if info:
            qty_step = Decimal(str(info['qty_step']))
            qty_decimal = Decimal(str(quantity))
            
            # Округляем до nearest step
            steps_count = (qty_decimal / qty_step).to_integral_value(rounding=ROUND_DOWN)
            rounded_qty = steps_count * qty_step
            
            # Определяем точность на основе qty_step
            # Если qty_step >= 1, используем целое число
            # Если qty_step < 1, определяем точность по количеству знаков после запятой
            if qty_step >= 1:
                # Для целых шагов (100, 1000 и т.д.) возвращаем целое число
                return f"{int(rounded_qty)}"
            else:
                # Для дробных шагов определяем точность
                step_str = format(info['qty_step'], 'f').rstrip('0')
                precision = len(step_str.split('.')[1]) if '.' in step_str else 0
                return f"{rounded_qty:.{precision}f}"
        return str(quantity)
    
    def execute(self, signal: Signal) -> bool:
        """Исполнить сигнал"""
        
        logger.debug(f"🎯 EXECUTE: {signal.action} {signal.symbol} @ {signal.price}, SL={signal.stop_loss}, TP={signal.take_profit}")
        
        # Проверяем лимит позиций
        positions = self.dm.get_positions()
        logger.debug(f"📊 Current positions: {len(positions)}")
        
        if len(positions) >= config.MAX_POSITIONS:
            logger.warning("Max positions reached")
            return False
        
        # Проверяем дубликаты
        for pos in positions:
            if pos['symbol'] == signal.symbol:
                logger.warning(f"Already have position in {signal.symbol}")
                return False
        
        try:
            self.set_leverage(signal.symbol)
            
            quantity = self.calculate_position_size(
                signal.symbol, 
                signal.price, 
                signal.stop_loss
            )
            
            if quantity <= 0:
                logger.error("Invalid position size")
                return False
            
            side = "Buy" if signal.action == "BUY" else "Sell"
            
            sl_price = self.round_price(signal.symbol, signal.stop_loss)
            tp_price = self.round_price(signal.symbol, signal.take_profit)
            qty_str = self.round_qty(signal.symbol, quantity)
            
            logger.debug(f"📤 API Request: symbol={signal.symbol}, side={side}, qty={qty_str}, sl={sl_price}, tp={tp_price}")
            
            order = self.dm.client.place_order(
                category="linear",
                symbol=signal.symbol,
                side=side,
                orderType="Market",
                qty=qty_str,
                timeInForce="GTC",
                stopLoss=sl_price,
                takeProfit=tp_price,
            )
            
            logger.debug(f"📥 API Response: {order}")
            
            # Проверяем ответ API
            if order and order.get('retCode') == 0:
                logger.info(
                    f"✅ ORDER PLACED: {signal.action} {signal.symbol}\n"
                    f"   Qty: {qty_str}\n"
                    f"   SL: {sl_price} | TP: {tp_price}\n"
                    f"   OrderId: {order.get('result', {}).get('orderId', 'N/A')}\n"
                    f"   Reason: {signal.reason}"
                )
            else:
                logger.error(
                    f"❌ ORDER FAILED: {signal.action} {signal.symbol}\n"
                    f"   Error: {order.get('retMsg', 'Unknown error')}\n"
                    f"   Response: {order}"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Order error: {e}")
            return False