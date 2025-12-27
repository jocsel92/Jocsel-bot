"""
Trading Bot - Core trading functionality
"""
import time
from datetime import datetime
from typing import Dict, List, Optional


class TradingBot:
    """
    A simple trading bot that can execute buy and sell orders.
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        """
        Initialize the trading bot.
        
        Args:
            initial_balance: Starting balance in USD
        """
        self.balance = initial_balance
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.trade_history: List[Dict] = []
        
    def get_balance(self) -> float:
        """Get current balance."""
        return self.balance
    
    def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        return self.positions.copy()
    
    def buy(self, symbol: str, quantity: float, price: float) -> bool:
        """
        Execute a buy order.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            quantity: Amount to buy
            price: Price per unit
            
        Returns:
            True if order successful, False otherwise
        """
        cost = quantity * price
        
        if cost > self.balance:
            print(f"Insufficient balance. Need ${cost:.2f}, have ${self.balance:.2f}")
            return False
        
        self.balance -= cost
        self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        
        trade = {
            'type': 'BUY',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'cost': cost,
            'timestamp': datetime.now().isoformat()
        }
        self.trade_history.append(trade)
        
        print(f"BUY: {quantity} {symbol} @ ${price:.2f} = ${cost:.2f}")
        return True
    
    def sell(self, symbol: str, quantity: float, price: float) -> bool:
        """
        Execute a sell order.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            quantity: Amount to sell
            price: Price per unit
            
        Returns:
            True if order successful, False otherwise
        """
        if symbol not in self.positions:
            print(f"No position in {symbol}")
            return False
        
        if self.positions[symbol] < quantity:
            print(f"Insufficient {symbol}. Need {quantity}, have {self.positions[symbol]}")
            return False
        
        revenue = quantity * price
        self.balance += revenue
        self.positions[symbol] -= quantity
        
        if self.positions[symbol] == 0:
            del self.positions[symbol]
        
        trade = {
            'type': 'SELL',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'revenue': revenue,
            'timestamp': datetime.now().isoformat()
        }
        self.trade_history.append(trade)
        
        print(f"SELL: {quantity} {symbol} @ ${price:.2f} = ${revenue:.2f}")
        return True
    
    def get_trade_history(self) -> List[Dict]:
        """Get all trade history."""
        return self.trade_history.copy()
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            prices: Current prices for each symbol
            
        Returns:
            Total portfolio value (balance + positions value)
        """
        positions_value = sum(
            quantity * prices.get(symbol, 0)
            for symbol, quantity in self.positions.items()
        )
        return self.balance + positions_value
