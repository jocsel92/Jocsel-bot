"""
Unit tests for the trading bot
"""
import unittest
from trading_bot import TradingBot
from config import Config


class TestTradingBot(unittest.TestCase):
    """Test cases for TradingBot class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.bot = TradingBot(initial_balance=10000.0)
    
    def test_initial_balance(self):
        """Test initial balance is set correctly."""
        self.assertEqual(self.bot.get_balance(), 10000.0)
    
    def test_initial_positions(self):
        """Test initial positions are empty."""
        self.assertEqual(self.bot.get_positions(), {})
    
    def test_buy_success(self):
        """Test successful buy order."""
        result = self.bot.buy('BTC', 0.1, 50000.0)
        self.assertTrue(result)
        self.assertEqual(self.bot.get_balance(), 10000.0 - 5000.0)
        self.assertEqual(self.bot.get_positions()['BTC'], 0.1)
    
    def test_buy_insufficient_balance(self):
        """Test buy fails with insufficient balance."""
        result = self.bot.buy('BTC', 1.0, 50000.0)
        self.assertFalse(result)
        self.assertEqual(self.bot.get_balance(), 10000.0)
        self.assertEqual(self.bot.get_positions(), {})
    
    def test_sell_success(self):
        """Test successful sell order."""
        self.bot.buy('BTC', 0.1, 50000.0)
        result = self.bot.sell('BTC', 0.05, 52000.0)
        self.assertTrue(result)
        self.assertEqual(self.bot.get_positions()['BTC'], 0.05)
        expected_balance = 10000.0 - 5000.0 + 2600.0
        self.assertEqual(self.bot.get_balance(), expected_balance)
    
    def test_sell_no_position(self):
        """Test sell fails when no position exists."""
        result = self.bot.sell('BTC', 0.5, 50000.0)
        self.assertFalse(result)
    
    def test_sell_insufficient_quantity(self):
        """Test sell fails with insufficient quantity."""
        self.bot.buy('BTC', 0.1, 50000.0)
        result = self.bot.sell('BTC', 0.5, 50000.0)
        self.assertFalse(result)
        self.assertEqual(self.bot.get_positions()['BTC'], 0.1)
    
    def test_sell_all_removes_position(self):
        """Test selling all quantity removes position."""
        self.bot.buy('BTC', 0.1, 50000.0)
        self.bot.sell('BTC', 0.1, 50000.0)
        self.assertNotIn('BTC', self.bot.get_positions())
    
    def test_multiple_buys_accumulate(self):
        """Test multiple buys accumulate quantity."""
        self.bot.buy('BTC', 0.05, 50000.0)
        self.bot.buy('BTC', 0.05, 50000.0)
        self.assertEqual(self.bot.get_positions()['BTC'], 0.1)
    
    def test_trade_history(self):
        """Test trade history is recorded."""
        self.bot.buy('BTC', 0.1, 50000.0)
        self.bot.sell('BTC', 0.05, 52000.0)
        history = self.bot.get_trade_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['type'], 'BUY')
        self.assertEqual(history[1]['type'], 'SELL')
    
    def test_portfolio_value(self):
        """Test portfolio value calculation."""
        self.bot.buy('BTC', 0.1, 50000.0)
        self.bot.buy('ETH', 1.0, 3000.0)
        
        current_prices = {'BTC': 52000.0, 'ETH': 3100.0}
        portfolio_value = self.bot.get_portfolio_value(current_prices)
        
        expected_balance = 10000.0 - 5000.0 - 3000.0  # -8000 total spent
        expected_positions_value = (0.1 * 52000.0) + (1.0 * 3100.0)  # 5200 + 3100
        expected_portfolio = expected_balance + expected_positions_value
        
        self.assertEqual(portfolio_value, expected_portfolio)


class TestConfig(unittest.TestCase):
    """Test cases for Config class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        self.assertEqual(config.get('initial_balance'), 10000.0)
        self.assertIn('BTC', config.get('symbols'))
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = Config(initial_balance=5000.0)
        self.assertEqual(config.get('initial_balance'), 5000.0)
    
    def test_set_config(self):
        """Test setting configuration values."""
        config = Config()
        config.set('new_key', 'new_value')
        self.assertEqual(config.get('new_key'), 'new_value')
    
    def test_get_all_config(self):
        """Test getting all configuration."""
        config = Config()
        all_config = config.get_all()
        self.assertIsInstance(all_config, dict)
        self.assertIn('initial_balance', all_config)


if __name__ == '__main__':
    unittest.main()
