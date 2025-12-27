#!/usr/bin/env python3
"""
Main entry point for Jocsel Trading Bot
"""
import argparse
from trading_bot import TradingBot
from config import Config


def main():
    """Main function to run the trading bot."""
    parser = argparse.ArgumentParser(
        description='Jocsel Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                    # Run basic demo
  python main.py --strategy                # Run strategy backtesting
  python strategy_runner.py --strategy all # Run all strategies
        """
    )
    parser.add_argument('--balance', type=float, default=10000.0,
                       help='Initial balance (default: 10000.0)')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo trading sequence')
    parser.add_argument('--strategy', action='store_true',
                       help='Run strategy backtesting')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config(initial_balance=args.balance)
    
    # Create trading bot
    bot = TradingBot(initial_balance=config.get('initial_balance'))
    
    print("=" * 60)
    print("Jocsel Trading Bot")
    print("=" * 60)
    print(f"Initial Balance: ${bot.get_balance():.2f}")
    print()
    
    if args.strategy:
        run_strategy_demo()
    elif args.demo:
        run_demo(bot)
    else:
        print("Bot initialized.")
        print("Use --demo to run a demo trading sequence.")
        print("Use --strategy to run strategy backtesting.")
        print(f"\nCurrent Balance: ${bot.get_balance():.2f}")
        print(f"Current Positions: {bot.get_positions()}")
    
    print()
    print("=" * 60)


def run_demo(bot: TradingBot):
    """Run a demo trading sequence."""
    print("Running demo trading sequence...")
    print()
    
    # Demo: Buy some assets
    print("--- Buying Assets ---")
    bot.buy('BTC', 0.05, 50000.0)  # Buy 0.05 BTC at $50,000 = $2,500
    bot.buy('ETH', 1.5, 3000.0)    # Buy 1.5 ETH at $3,000 = $4,500
    bot.buy('SOL', 10.0, 100.0)    # Buy 10 SOL at $100 = $1,000
    print()
    
    print(f"Balance after purchases: ${bot.get_balance():.2f}")
    print(f"Current Positions: {bot.get_positions()}")
    print()
    
    # Demo: Sell some assets
    print("--- Selling Assets ---")
    bot.sell('BTC', 0.02, 52000.0)  # Sell 0.02 BTC at $52,000 (profit)
    bot.sell('ETH', 0.5, 2900.0)    # Sell 0.5 ETH at $2,900 (loss)
    print()
    
    print(f"Balance after sales: ${bot.get_balance():.2f}")
    print(f"Current Positions: {bot.get_positions()}")
    print()
    
    # Calculate portfolio value
    current_prices = {'BTC': 51000.0, 'ETH': 3100.0, 'SOL': 105.0}
    portfolio_value = bot.get_portfolio_value(current_prices)
    print(f"Total Portfolio Value: ${portfolio_value:.2f}")
    
    pnl = portfolio_value - 10000.0
    pnl_pct = (pnl / 10000.0) * 100
    print(f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
    print()
    
    # Show trade history
    print("--- Trade History ---")
    for trade in bot.get_trade_history():
        trade_type = trade['type']
        symbol = trade['symbol']
        quantity = trade['quantity']
        price = trade['price']
        print(f"{trade_type}: {quantity} {symbol} @ ${price:.2f}")


def run_strategy_demo():
    """Run a quick strategy demonstration."""
    from market_data import MarketData
    from strategy import MovingAverageCrossover
    from backtesting import Backtester
    
    print("Running strategy backtesting demo...")
    print()
    
    # Generate market data
    market = MarketData('BTC', 50000.0)
    market.generate_price_data(num_periods=100, volatility=0.02)
    
    # Test Moving Average strategy
    strategy = MovingAverageCrossover(short_period=10, long_period=30)
    backtester = Backtester(initial_balance=10000.0)
    
    results = backtester.run_backtest(
        strategy=strategy,
        market_data=market,
        symbol='BTC',
        position_size=0.2
    )
    
    backtester.print_results()
    
    print("\nFor more detailed strategy testing, use:")
    print("  python strategy_runner.py --strategy all")


if __name__ == '__main__':
    main()
