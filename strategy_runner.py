#!/usr/bin/env python3
"""
Strategy runner - Execute trading strategies with backtesting
"""
import argparse
from market_data import MarketData
from strategy import MovingAverageCrossover, RSIStrategy, MomentumStrategy
from backtesting import Backtester


def run_strategy(strategy_name: str, symbol: str = 'BTC', 
                initial_balance: float = 10000.0, periods: int = 200):
    """
    Run a trading strategy with backtesting.
    
    Args:
        strategy_name: Name of strategy ('ma', 'rsi', 'momentum', 'all')
        symbol: Trading symbol
        initial_balance: Starting balance
        periods: Number of periods for backtesting
    """
    print("\n" + "=" * 60)
    print("JOCSEL TRADING BOT - STRATEGY RUNNER")
    print("=" * 60)
    
    # Generate market data
    print(f"\nGenerating market data for {symbol}...")
    market = MarketData(symbol=symbol, initial_price=50000.0)
    market.generate_price_data(num_periods=periods, volatility=0.02)
    print(f"Generated {periods} periods of price data")
    
    strategies = []
    
    if strategy_name == 'ma' or strategy_name == 'all':
        strategies.append(MovingAverageCrossover(short_period=10, long_period=30))
    
    if strategy_name == 'rsi' or strategy_name == 'all':
        strategies.append(RSIStrategy(period=14, oversold=30, overbought=70))
    
    if strategy_name == 'momentum' or strategy_name == 'all':
        strategies.append(MomentumStrategy(lookback=5))
    
    if not strategies:
        print(f"Unknown strategy: {strategy_name}")
        print("Available strategies: ma, rsi, momentum, all")
        return
    
    # Run backtests
    best_strategy = None
    best_return = float('-inf')
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing: {strategy.name}")
        print(f"{'='*60}")
        
        backtester = Backtester(initial_balance=initial_balance)
        results = backtester.run_backtest(
            strategy=strategy,
            market_data=market,
            symbol=symbol,
            position_size=0.2  # Use 20% of portfolio per trade
        )
        
        backtester.print_results()
        
        # Track best performing strategy
        if results['return_pct'] > best_return:
            best_return = results['return_pct']
            best_strategy = strategy.name
    
    # Summary
    if len(strategies) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Best Strategy: {best_strategy}")
        print(f"Best Return: {best_return:+.2f}%")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Jocsel Trading Bot - Strategy Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python strategy_runner.py --strategy ma
  python strategy_runner.py --strategy rsi --balance 50000
  python strategy_runner.py --strategy all --periods 500
  python strategy_runner.py --strategy momentum --symbol ETH
        """
    )
    
    parser.add_argument('--strategy', type=str, default='all',
                       choices=['ma', 'rsi', 'momentum', 'all'],
                       help='Trading strategy to run (default: all)')
    parser.add_argument('--symbol', type=str, default='BTC',
                       help='Trading symbol (default: BTC)')
    parser.add_argument('--balance', type=float, default=10000.0,
                       help='Initial balance (default: 10000.0)')
    parser.add_argument('--periods', type=int, default=200,
                       help='Number of periods for backtesting (default: 200)')
    
    args = parser.parse_args()
    
    run_strategy(
        strategy_name=args.strategy,
        symbol=args.symbol,
        initial_balance=args.balance,
        periods=args.periods
    )
    
    print("\n" + "=" * 60)
    print("Backtesting complete!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
