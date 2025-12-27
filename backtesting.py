"""
Backtesting engine for trading strategies
"""
from typing import List, Dict
from trading_bot import TradingBot
from strategy import TradingStrategy, Signal
from market_data import MarketData


class Backtester:
    """Backtesting engine for evaluating trading strategies."""
    
    def __init__(self, initial_balance: float = 10000.0):
        """
        Initialize backtester.
        
        Args:
            initial_balance: Starting balance for backtest
        """
        self.initial_balance = initial_balance
        self.results: Dict = {}
    
    def run_backtest(self, strategy: TradingStrategy, market_data: MarketData, 
                     symbol: str, position_size: float = 0.1) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            strategy: Trading strategy to test
            market_data: Market data object with price history
            symbol: Trading symbol
            position_size: Fraction of portfolio to use per trade (0.1 = 10%)
            
        Returns:
            Dictionary with backtest results and metrics
        """
        bot = TradingBot(initial_balance=self.initial_balance)
        price_history = market_data.get_price_history()
        
        trades_executed = 0
        winning_trades = 0
        losing_trades = 0
        
        # Track cost basis for position (FIFO)
        position_queue = []  # List of (quantity, price) tuples
        
        # Iterate through price history
        for i in range(len(price_history)):
            current_data = price_history[:i+1]
            current_price = current_data[-1]['close']
            
            # Generate signal
            signal = strategy.generate_signal(current_data)
            
            # Execute trades based on signal
            if signal == Signal.BUY and bot.get_balance() > 0:
                # Calculate quantity based on position size
                max_cost = bot.get_balance() * position_size
                quantity = max_cost / current_price
                
                if bot.buy(symbol, quantity, current_price):
                    trades_executed += 1
                    # Track cost basis for this purchase
                    position_queue.append((quantity, current_price))
            
            elif signal == Signal.SELL and symbol in bot.get_positions():
                # Sell all position
                quantity_to_sell = bot.get_positions()[symbol]
                
                if bot.sell(symbol, quantity_to_sell, current_price):
                    trades_executed += 1
                    
                    # Calculate P&L using FIFO cost basis
                    remaining = quantity_to_sell
                    total_cost = 0
                    while remaining > 0 and position_queue:
                        qty, price = position_queue.pop(0)
                        if qty <= remaining:
                            total_cost += qty * price
                            remaining -= qty
                        else:
                            total_cost += remaining * price
                            position_queue.insert(0, (qty - remaining, price))
                            remaining = 0
                    
                    # Track winning/losing trades
                    avg_cost = total_cost / quantity_to_sell if quantity_to_sell > 0 else 0
                    if current_price > avg_cost:
                        winning_trades += 1
                    elif current_price < avg_cost:
                        losing_trades += 1
        
        # Calculate final portfolio value
        final_prices = {symbol: price_history[-1]['close']}
        final_value = bot.get_portfolio_value(final_prices)
        
        # Calculate metrics
        total_return = final_value - self.initial_balance
        return_pct = (total_return / self.initial_balance) * 100
        win_rate = (winning_trades / trades_executed * 100) if trades_executed > 0 else 0
        
        results = {
            'strategy': strategy.name,
            'initial_balance': self.initial_balance,
            'final_value': final_value,
            'total_return': total_return,
            'return_pct': return_pct,
            'trades_executed': trades_executed,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'trade_history': bot.get_trade_history()
        }
        
        self.results = results
        return results
    
    def print_results(self):
        """Print formatted backtest results."""
        if not self.results:
            print("No backtest results available.")
            return
        
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Strategy: {self.results['strategy']}")
        print(f"Initial Balance: ${self.results['initial_balance']:,.2f}")
        print(f"Final Value: ${self.results['final_value']:,.2f}")
        print(f"Total Return: ${self.results['total_return']:,.2f} ({self.results['return_pct']:+.2f}%)")
        print(f"\nTrades Executed: {self.results['trades_executed']}")
        print(f"Winning Trades: {self.results['winning_trades']}")
        print(f"Losing Trades: {self.results['losing_trades']}")
        print(f"Win Rate: {self.results['win_rate']:.1f}%")
        print("=" * 60)
