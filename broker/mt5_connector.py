"""
MT5 Connector — MetaTrader 5 connection, order execution, position management.
===============================================================================
"""

import MetaTrader5 as mt5
import numpy as np

from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MAGIC


def connect() -> bool:
    """Initialize and log in to MT5."""
    if not mt5.initialize():
        print(f"❌ MT5 initialize() failed: {mt5.last_error()}")
        return False

    if not mt5.login(MT5_LOGIN, ****** server=MT5_SERVER):
        print(f"❌ MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False

    info = mt5.account_info()
    print(
        f"✅ MT5 conectado — cuenta {info.login}, "
        f"balance {info.balance}, server {info.server}"
    )
    return True


def shutdown():
    """Shut down MT5 connection."""
    mt5.shutdown()
    print("MT5 desconectado")


def symbol_info(symbol: str):
    """Return symbol info or None."""
    info = mt5.symbol_info(symbol)
    if info is None:
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)
    return info


def pip_value(symbol: str) -> float:
    """Return pip size for a symbol (e.g. 0.0001 for EURUSD)."""
    info = symbol_info(symbol)
    if info is None:
        return 0.0001
    return 10 ** (-info.digits) * 10  # 1 pip = 10 points for 5-digit


def price_text(symbol: str, value: float) -> str:
    """Format price with correct digits."""
    info = symbol_info(symbol)
    digits = info.digits if info else 5
    return f"{float(value):.{digits}f}"


def open_order(symbol: str, action: str, lot: float,
               sl_price: float, tp_price: float):
    """
    Send a market order to MT5.

    Parameters
    ----------
    symbol : str
    action : 'buy' or 'sell'
    lot : float
    sl_price : float
    tp_price : float

    Returns
    -------
    result or None
    """
    info = symbol_info(symbol)
    if info is None:
        print(f"❌ Símbolo {symbol} no disponible")
        return None

    order_type = mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL
    price = info.ask if action == "buy" else info.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot),
        "type": order_type,
        "price": price,
        "sl": float(sl_price),
        "tp": float(tp_price),
        "deviation": 20,
        "magic": MAGIC,
        "comment": "jocsel_bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        error = result.retcode if result else "None"
        print(f"❌ Orden fallida {symbol} {action}: retcode={error}")
        return None

    print(
        f"✅ Orden ejecutada {symbol} {action} lot={lot} "
        f"price={result.price} sl={sl_price} tp={tp_price}"
    )
    return result


def modify_position(ticket: int, symbol: str,
                    new_sl: float = None, new_tp: float = None):
    """Modify SL/TP of an open position."""
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return None

    pos = positions[0]
    sl = new_sl if new_sl is not None else pos.sl
    tp = new_tp if new_tp is not None else pos.tp

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol": symbol,
        "sl": float(sl),
        "tp": float(tp),
        "magic": MAGIC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return result
    return None


def get_open_positions(symbol: str = None):
    """Return open positions, optionally filtered by symbol."""
    if symbol:
        positions = mt5.positions_get(symbol=symbol)
    else:
        positions = mt5.positions_get()

    if positions is None:
        return []
    return [p for p in positions if p.magic == MAGIC]


def get_rates(symbol: str, timeframe, count: int = 250):
    """Get OHLCV rates from MT5."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        return None
    import pandas as pd
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    return df


def history_deals_get(start, end):
    """Get deal history in a time range."""
    return mt5.history_deals_get(start, end)
