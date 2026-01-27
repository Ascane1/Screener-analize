from typing import List, Tuple
from trading_bot.data_manager import DataManager
from trading_bot.config import config
import logging

logger = logging.getLogger(__name__)


class Screener:
    """Отбор топ монет по волатильности"""
    
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager
    
    def get_top_movers(self) -> Tuple[List[dict], List[dict]]:
        """
        Получить топ монет по росту и падению
        Returns: (gainers, losers)
        """
        tickers = self.dm.get_all_tickers()
        
        if not tickers:
            return [], []
        
        # Символы с открытыми позициями
        positions = self.dm.get_positions()
        position_symbols = {p['symbol'] for p in positions}
        
        # Фильтруем
        filtered = []
        for symbol, data in tickers.items():
            if symbol in config.EXCLUDED_SYMBOLS:
                continue
            if symbol in position_symbols:
                continue
            if data['volume_24h'] < config.MIN_VOLUME_24H:
                continue
            
            filtered.append(data)
        
        # Сортируем по изменению цены
        sorted_by_change = sorted(
            filtered, 
            key=lambda x: x['price_change_24h']
        )
        
        # Топ падающих
        losers = [
            t for t in sorted_by_change[:config.TOP_LOSERS_COUNT]
            if t['price_change_24h'] <= -config.MIN_PRICE_CHANGE
        ]
        
        # Топ растущих
        gainers = [
            t for t in sorted_by_change[-config.TOP_GAINERS_COUNT:][::-1]
            if t['price_change_24h'] >= config.MIN_PRICE_CHANGE
        ]
        
        logger.info(f"Screener: {len(gainers)} gainers, {len(losers)} losers")
        
        for g in gainers[:3]:
            logger.info(f"  📈 {g['symbol']}: +{g['price_change_24h']:.2f}%")
        for l in losers[:3]:
            logger.info(f"  📉 {l['symbol']}: {l['price_change_24h']:.2f}%")
        
        return gainers, losers
    
    def get_symbols_to_analyze(self) -> List[str]:
        """Получить список символов для анализа"""
        gainers, losers = self.get_top_movers()
        return [item['symbol'] for item in gainers + losers]