"""
Market data simulator for backtesting and live trading
"""
import random
from typing import List, Dict
from datetime import datetime, timedelta


class MarketData:
    """Simulates market price data for testing trading strategies."""
    
    def __init__(self, symbol: str, initial_price: float = 50000.0):
        """
        Initialize market data simulator.
        
        Args:
            symbol: Trading symbol
            initial_price: Starting price
        """
        self.symbol = symbol
        self.initial_price = initial_price
        self.current_price = initial_price
        self.price_history: List[Dict] = []
        
    def generate_price_data(self, num_periods: int = 100, volatility: float = 0.02) -> List[Dict]:
        """
        Generate synthetic price data with random walk.
        
        Args:
            num_periods: Number of time periods to generate
            volatility: Price volatility (0.02 = 2% per period)
            
        Returns:
            List of price dictionaries with timestamp, open, high, low, close
        """
        prices = []
        current_price = self.initial_price
        start_time = datetime.now() - timedelta(hours=num_periods)
        
        for i in range(num_periods):
            # Random walk with drift
            change = random.gauss(0.0005, volatility)  # Slight upward drift
            current_price = current_price * (1 + change)
            
            # Generate OHLC data
            open_price = current_price
            high_price = current_price * (1 + abs(random.gauss(0, volatility/2)))
            low_price = current_price * (1 - abs(random.gauss(0, volatility/2)))
            close_price = random.uniform(low_price, high_price)
            
            price_data = {
                'timestamp': start_time + timedelta(hours=i),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': random.uniform(100, 1000)
            }
            prices.append(price_data)
            current_price = close_price
        
        self.price_history = prices
        self.current_price = current_price
        return prices
    
    def get_latest_price(self) -> float:
        """Get the most recent price."""
        if self.price_history:
            return self.price_history[-1]['close']
        return self.current_price
    
    def get_price_history(self) -> List[Dict]:
        """Get all historical price data."""
        return self.price_history.copy()
