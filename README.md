# Jocsel-bot

A simple Python trading bot that can execute buy and sell orders with portfolio management.

## Features

- Execute buy and sell orders for different trading symbols
- Track portfolio balance and positions
- Maintain trade history
- Calculate portfolio value
- Simple configuration management

## Installation

No external dependencies required! The bot uses only Python's standard library.

```bash
git clone https://github.com/jocsel92/Jocsel-bot.git
cd Jocsel-bot
```

## Usage

### Basic Usage

Run the bot with default settings:

```bash
python main.py
```

### Demo Mode

Run a demo trading sequence:

```bash
python main.py --demo
```

### Custom Initial Balance

Start with a custom balance:

```bash
python main.py --balance 50000 --demo
```

## Using the Trading Bot Programmatically

```python
from trading_bot import TradingBot

# Create a bot with $10,000 initial balance
bot = TradingBot(initial_balance=10000.0)

# Buy 0.5 BTC at $50,000
bot.buy('BTC', 0.5, 50000.0)

# Sell 0.2 BTC at $52,000
bot.sell('BTC', 0.2, 52000.0)

# Check balance and positions
print(f"Balance: ${bot.get_balance():.2f}")
print(f"Positions: {bot.get_positions()}")

# Calculate portfolio value
current_prices = {'BTC': 51000.0}
portfolio_value = bot.get_portfolio_value(current_prices)
print(f"Portfolio Value: ${portfolio_value:.2f}")
```

## Running Tests

Run the test suite:

```bash
python -m unittest test_trading_bot.py
```

## API Reference

### TradingBot

- `__init__(initial_balance: float = 10000.0)` - Initialize the bot
- `buy(symbol: str, quantity: float, price: float) -> bool` - Execute a buy order
- `sell(symbol: str, quantity: float, price: float) -> bool` - Execute a sell order
- `get_balance() -> float` - Get current balance
- `get_positions() -> Dict[str, float]` - Get current positions
- `get_trade_history() -> List[Dict]` - Get all trade history
- `get_portfolio_value(prices: Dict[str, float]) -> float` - Calculate total portfolio value

## License

MIT License