"""
Trading strategies implementation
"""
from typing import List, Dict, Optional
from enum import Enum


class Signal(Enum):
    """Trading signals"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradingStrategy:
    """Base class for trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    def generate_signal(self, price_history: List[Dict]) -> Signal:
        """
        Generate trading signal based on price history.
        
        Args:
            price_history: List of price data dictionaries
            
        Returns:
            Trading signal (BUY, SELL, or HOLD)
        """
        raise NotImplementedError("Subclasses must implement generate_signal")


class MovingAverageCrossover(TradingStrategy):
    """
    Moving Average Crossover Strategy.
    
    Generates BUY signal when short MA crosses above long MA.
    Generates SELL signal when short MA crosses below long MA.
    """
    
    def __init__(self, short_period: int = 10, long_period: int = 30):
        super().__init__("MA Crossover")
        self.short_period = short_period
        self.long_period = long_period
        self.last_signal = Signal.HOLD
    
    def calculate_ma(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate moving average."""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period
    
    def generate_signal(self, price_history: List[Dict]) -> Signal:
        """Generate signal based on MA crossover."""
        if len(price_history) < self.long_period:
            return Signal.HOLD
        
        closes = [p['close'] for p in price_history]
        
        short_ma = self.calculate_ma(closes, self.short_period)
        long_ma = self.calculate_ma(closes, self.long_period)
        
        if short_ma is None or long_ma is None:
            return Signal.HOLD
        
        # Previous MAs for crossover detection
        prev_closes = closes[:-1]
        prev_short_ma = self.calculate_ma(prev_closes, self.short_period)
        prev_long_ma = self.calculate_ma(prev_closes, self.long_period)
        
        if prev_short_ma is None or prev_long_ma is None:
            return Signal.HOLD
        
        # Bullish crossover: short MA crosses above long MA
        if prev_short_ma <= prev_long_ma and short_ma > long_ma:
            self.last_signal = Signal.BUY
            return Signal.BUY
        
        # Bearish crossover: short MA crosses below long MA
        if prev_short_ma >= prev_long_ma and short_ma < long_ma:
            self.last_signal = Signal.SELL
            return Signal.SELL
        
        return Signal.HOLD


class RSIStrategy(TradingStrategy):
    """
    RSI (Relative Strength Index) Strategy.
    
    Generates BUY signal when RSI < oversold threshold.
    Generates SELL signal when RSI > overbought threshold.
    """
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__("RSI Strategy")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, prices: List[float]) -> Optional[float]:
        """Calculate RSI indicator."""
        if len(prices) < self.period + 1:
            return None
        
        # Calculate price changes
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Separate gains and losses
        gains = [max(c, 0) for c in changes[-self.period:]]
        losses = [abs(min(c, 0)) for c in changes[-self.period:]]
        
        avg_gain = sum(gains) / self.period
        avg_loss = sum(losses) / self.period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signal(self, price_history: List[Dict]) -> Signal:
        """Generate signal based on RSI."""
        if len(price_history) < self.period + 1:
            return Signal.HOLD
        
        closes = [p['close'] for p in price_history]
        rsi = self.calculate_rsi(closes)
        
        if rsi is None:
            return Signal.HOLD
        
        if rsi < self.oversold:
            return Signal.BUY
        elif rsi > self.overbought:
            return Signal.SELL
        
        return Signal.HOLD


class MomentumStrategy(TradingStrategy):
    """
    Simple Momentum Strategy.
    
    Buys when price momentum is positive.
    Sells when price momentum turns negative.
    """
    
    def __init__(self, lookback: int = 5):
        super().__init__("Momentum Strategy")
        self.lookback = lookback
    
    def generate_signal(self, price_history: List[Dict]) -> Signal:
        """Generate signal based on momentum."""
        if len(price_history) < self.lookback + 1:
            return Signal.HOLD
        
        closes = [p['close'] for p in price_history]
        
        # Calculate momentum
        current_price = closes[-1]
        past_price = closes[-self.lookback]
        momentum = (current_price - past_price) / past_price
        
        # Strong positive momentum
        if momentum > 0.02:  # 2% gain
            return Signal.BUY
        # Strong negative momentum
        elif momentum < -0.02:  # 2% loss
            return Signal.SELL
        
        return Signal.HOLD
