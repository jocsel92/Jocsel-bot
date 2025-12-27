# Jocsel-bot

Un bot de trading completo en Python con ejecución de órdenes de compra/venta, estrategias automatizadas y capacidades de backtesting.

## Características

### Bot de Trading Base
- Ejecutar órdenes de compra y venta para diferentes símbolos
- Rastrear saldo de cartera y posiciones
- Mantener historial de operaciones
- Calcular valor de cartera
- Sistema de gestión de configuración

### Estrategias de Trading Automatizadas
- **Moving Average Crossover**: Señales basadas en cruce de medias móviles
- **RSI Strategy**: Compra en sobreventa (RSI < 30), vende en sobrecompra (RSI > 70)
- **Momentum Strategy**: Opera basándose en el momentum de precios
- Arquitectura extensible para agregar nuevas estrategias

### Motor de Backtesting
- Probar estrategias en datos históricos
- Métricas detalladas de rendimiento
- Cálculo de tasa de victorias y retornos
- Simulación de datos de mercado
- Gestión automática de posiciones

## Instalación

No requiere dependencias externas. El bot usa solo la biblioteca estándar de Python.

```bash
git clone https://github.com/jocsel92/Jocsel-bot.git
cd Jocsel-bot
```

## Uso

### Demo Básico

Ejecutar secuencia de trading de demostración:

```bash
python main.py --demo
```

### Backtesting de Estrategias

Ejecutar demo rápido de backtesting:

```bash
python main.py --strategy
```

### Ejecutor de Estrategias Completo

Probar todas las estrategias:

```bash
python strategy_runner.py --strategy all
```

Probar estrategia específica:

```bash
python strategy_runner.py --strategy ma
python strategy_runner.py --strategy rsi
python strategy_runner.py --strategy momentum
```

Opciones avanzadas:

```bash
# Con balance personalizado
python strategy_runner.py --strategy all --balance 50000

# Más períodos de backtesting
python strategy_runner.py --strategy all --periods 500

# Símbolo diferente
python strategy_runner.py --strategy rsi --symbol ETH
```

## Uso Programático

### Bot de Trading Básico

```python
from trading_bot import TradingBot

# Crear bot con $10,000 de balance inicial
bot = TradingBot(initial_balance=10000.0)

# Comprar 0.5 BTC a $50,000
bot.buy('BTC', 0.5, 50000.0)

# Vender 0.2 BTC a $52,000
bot.sell('BTC', 0.2, 52000.0)

# Verificar balance y posiciones
print(f"Balance: ${bot.get_balance():.2f}")
print(f"Posiciones: {bot.get_positions()}")

# Calcular valor de cartera
current_prices = {'BTC': 51000.0}
portfolio_value = bot.get_portfolio_value(current_prices)
print(f"Valor de Cartera: ${portfolio_value:.2f}")
```

### Backtesting de Estrategias

```python
from market_data import MarketData
from strategy import MovingAverageCrossover, RSIStrategy
from backtesting import Backtester

# Generar datos de mercado
market = MarketData('BTC', initial_price=50000.0)
market.generate_price_data(num_periods=200, volatility=0.02)

# Crear estrategia
strategy = MovingAverageCrossover(short_period=10, long_period=30)

# Ejecutar backtest
backtester = Backtester(initial_balance=10000.0)
results = backtester.run_backtest(
    strategy=strategy,
    market_data=market,
    symbol='BTC',
    position_size=0.2  # Usar 20% de cartera por operación
)

# Mostrar resultados
backtester.print_results()
```

### Crear Estrategia Personalizada

```python
from strategy import TradingStrategy, Signal

class MyCustomStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("My Custom Strategy")
    
    def generate_signal(self, price_history):
        # Implementar tu lógica aquí
        if len(price_history) < 10:
            return Signal.HOLD
        
        # Ejemplo: comprar si el precio está subiendo
        recent_prices = [p['close'] for p in price_history[-10:]]
        if recent_prices[-1] > recent_prices[0]:
            return Signal.BUY
        elif recent_prices[-1] < recent_prices[0]:
            return Signal.SELL
        
        return Signal.HOLD
```

## Ejecutar Tests

Ejecutar todos los tests:

```bash
python -m unittest discover -v
```

Ejecutar tests específicos:

```bash
python -m unittest test_trading_bot.py
python -m unittest test_strategies.py
```

## Arquitectura

### Componentes Principales

- **`trading_bot.py`**: Motor de ejecución de trading central
- **`strategy.py`**: Implementaciones de estrategias de trading
- **`market_data.py`**: Simulador de datos de mercado
- **`backtesting.py`**: Motor de backtesting
- **`strategy_runner.py`**: CLI para ejecutar estrategias
- **`config.py`**: Gestión de configuración
- **`main.py`**: Punto de entrada principal

### Estrategias Incluidas

1. **Moving Average Crossover**
   - Compra cuando MA corta cruza por encima de MA larga
   - Vende cuando MA corta cruza por debajo de MA larga
   - Parámetros: short_period (10), long_period (30)

2. **RSI Strategy**
   - Compra cuando RSI < 30 (sobreventa)
   - Vende cuando RSI > 70 (sobrecompra)
   - Parámetros: period (14), oversold (30), overbought (70)

3. **Momentum Strategy**
   - Compra en momentum positivo fuerte (>2%)
   - Vende en momentum negativo fuerte (<-2%)
   - Parámetros: lookback (5)

## Próximos Pasos / Mejoras Potenciales

1. **Integración con APIs Reales**
   - Conectar con Binance, Coinbase, etc.
   - Feeds de precios en tiempo real
   - Ejecución de órdenes en vivo

2. **Estrategias Avanzadas**
   - Machine Learning
   - Análisis de sentimiento
   - Arbitraje
   - Market making

3. **Persistencia de Datos**
   - Guardar historial en base de datos
   - Rastreo de rendimiento a largo plazo
   - Análisis de datos históricos

4. **Gestión de Riesgo**
   - Stop-loss automático
   - Take-profit
   - Límites de posición
   - Diversificación de cartera

5. **Dashboard Web**
   - Interfaz gráfica
   - Gráficos en tiempo real
   - Control de estrategias
   - Monitoreo de rendimiento

## API Reference

### TradingBot

- `__init__(initial_balance: float = 10000.0)` - Inicializar el bot
- `buy(symbol: str, quantity: float, price: float) -> bool` - Ejecutar orden de compra
- `sell(symbol: str, quantity: float, price: float) -> bool` - Ejecutar orden de venta
- `get_balance() -> float` - Obtener balance actual
- `get_positions() -> Dict[str, float]` - Obtener posiciones actuales
- `get_trade_history() -> List[Dict]` - Obtener todo el historial de operaciones
- `get_portfolio_value(prices: Dict[str, float]) -> float` - Calcular valor total de cartera

### TradingStrategy

- `generate_signal(price_history: List[Dict]) -> Signal` - Generar señal de trading

### Backtester

- `run_backtest(strategy, market_data, symbol, position_size) -> Dict` - Ejecutar backtest
- `print_results()` - Imprimir resultados formateados

## Licencia

MIT License