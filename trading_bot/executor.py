import math
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
            quantity = math.floor(quantity / qty_step) * qty_step
            quantity = max(quantity, min_qty)
        
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
    
    def round_price(self, symbol: str, price: float) -> float:
        """Округлить цену до tick size"""
        info = self._get_instrument_info(symbol)
        if info:
            tick = info['tick_size']
            return round(price / tick) * tick
        return price
    
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
            
            logger.debug(f"📤 API Request: symbol={signal.symbol}, side={side}, qty={quantity}, sl={sl_price}, tp={tp_price}")
            
            order = self.dm.client.place_order(
                category="linear",
                symbol=signal.symbol,
                side=side,
                orderType="Market",
                qty=str(quantity),
                timeInForce="GTC",
                stopLoss=str(sl_price),
                takeProfit=str(tp_price),
            )
            
            logger.debug(f"📥 API Response: {order}")
            
            # Проверяем ответ API
            if order and order.get('retCode') == 0:
                logger.info(
                    f"✅ ORDER PLACED: {signal.action} {signal.symbol}\n"
                    f"   Qty: {quantity}\n"
                    f"   SL: {sl_price:.5f} | TP: {tp_price:.5f}\n"
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