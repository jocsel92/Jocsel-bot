"""
Tests for trading strategies and backtesting
"""
import unittest
from market_data import MarketData
from strategy import MovingAverageCrossover, RSIStrategy, MomentumStrategy, Signal
from backtesting import Backtester


class TestMarketData(unittest.TestCase):
    """Test cases for MarketData class."""
    
    def test_initialization(self):
        """Test market data initialization."""
        market = MarketData('BTC', 50000.0)
        self.assertEqual(market.symbol, 'BTC')
        self.assertEqual(market.initial_price, 50000.0)
    
    def test_generate_price_data(self):
        """Test price data generation."""
        market = MarketData('BTC', 50000.0)
        prices = market.generate_price_data(num_periods=100)
        self.assertEqual(len(prices), 100)
        self.assertIn('close', prices[0])
        self.assertIn('open', prices[0])
        self.assertIn('high', prices[0])
        self.assertIn('low', prices[0])
    
    def test_get_latest_price(self):
        """Test getting latest price."""
        market = MarketData('BTC', 50000.0)
        market.generate_price_data(num_periods=10)
        latest = market.get_latest_price()
        self.assertIsInstance(latest, float)
        self.assertGreater(latest, 0)


class TestMovingAverageCrossover(unittest.TestCase):
    """Test cases for MA Crossover strategy."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = MovingAverageCrossover(short_period=5, long_period=10)
        self.assertEqual(strategy.short_period, 5)
        self.assertEqual(strategy.long_period, 10)
    
    def test_signal_with_insufficient_data(self):
        """Test signal generation with insufficient data."""
        strategy = MovingAverageCrossover(short_period=5, long_period=10)
        market = MarketData('BTC', 50000.0)
        prices = market.generate_price_data(num_periods=5)
        signal = strategy.generate_signal(prices)
        self.assertEqual(signal, Signal.HOLD)
    
    def test_signal_generation(self):
        """Test signal generation with sufficient data."""
        strategy = MovingAverageCrossover(short_period=5, long_period=10)
        market = MarketData('BTC', 50000.0)
        prices = market.generate_price_data(num_periods=50)
        signal = strategy.generate_signal(prices)
        self.assertIn(signal, [Signal.BUY, Signal.SELL, Signal.HOLD])


class TestRSIStrategy(unittest.TestCase):
    """Test cases for RSI strategy."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = RSIStrategy(period=14, oversold=30, overbought=70)
        self.assertEqual(strategy.period, 14)
        self.assertEqual(strategy.oversold, 30)
        self.assertEqual(strategy.overbought, 70)
    
    def test_signal_generation(self):
        """Test RSI signal generation."""
        strategy = RSIStrategy(period=14)
        market = MarketData('BTC', 50000.0)
        prices = market.generate_price_data(num_periods=50)
        signal = strategy.generate_signal(prices)
        self.assertIn(signal, [Signal.BUY, Signal.SELL, Signal.HOLD])


class TestMomentumStrategy(unittest.TestCase):
    """Test cases for Momentum strategy."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = MomentumStrategy(lookback=5)
        self.assertEqual(strategy.lookback, 5)
    
    def test_signal_generation(self):
        """Test momentum signal generation."""
        strategy = MomentumStrategy(lookback=5)
        market = MarketData('BTC', 50000.0)
        prices = market.generate_price_data(num_periods=20)
        signal = strategy.generate_signal(prices)
        self.assertIn(signal, [Signal.BUY, Signal.SELL, Signal.HOLD])


class TestBacktester(unittest.TestCase):
    """Test cases for Backtester class."""
    
    def test_initialization(self):
        """Test backtester initialization."""
        backtester = Backtester(initial_balance=10000.0)
        self.assertEqual(backtester.initial_balance, 10000.0)
    
    def test_run_backtest(self):
        """Test running a backtest."""
        strategy = MovingAverageCrossover(short_period=5, long_period=10)
        market = MarketData('BTC', 50000.0)
        market.generate_price_data(num_periods=100)
        
        backtester = Backtester(initial_balance=10000.0)
        results = backtester.run_backtest(strategy, market, 'BTC', position_size=0.1)
        
        self.assertIn('strategy', results)
        self.assertIn('final_value', results)
        self.assertIn('total_return', results)
        self.assertIn('trades_executed', results)
        self.assertIsInstance(results['final_value'], float)
        self.assertGreaterEqual(results['trades_executed'], 0)


if __name__ == '__main__':
    unittest.main()
