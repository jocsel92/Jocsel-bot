"""
TradeLocker Connector — Copy trades from MT5 to TradeLocker.
=============================================================
"""

import os
from config import (
    TRADELOCKER_ENABLED,
    TRADELOCKER_EMAIL,
    TRADELOCKER_PASSWORD,
    TRADELOCKER_SERVER,
    TRADELOCKER_ENV,
    TRADELOCKER_ENV_URLS,
)

try:
    from tradelocker import TLAPI
    _TL_AVAILABLE = True
except ImportError:
    TLAPI = None
    _TL_AVAILABLE = False

_client = None
_instrument_cache: dict = {}


def init() -> bool:
    """Initialize TradeLocker API client."""
    global _client
    if not TRADELOCKER_ENABLED or not _TL_AVAILABLE:
        print("TradeLocker deshabilitado o librería no instalada")
        return False

    try:
        env_url = TRADELOCKER_ENV_URLS.get(
            TRADELOCKER_ENV.lower(),
            TRADELOCKER_ENV_URLS["demo"],
        )
        _client = TLAPI(
            environment=env_url,
            username=TRADELOCKER_EMAIL,
            ******
            server=TRADELOCKER_SERVER,
        )
        instruments = _client.get_all_instruments()
        print(f"✅ TradeLocker conectado — {len(instruments)} instrumentos")
        return True
    except Exception as e:
        print(f"❌ TradeLocker error: {e}")
        _client = None
        return False


def _get_instrument_id(symbol: str):
    """Resolve symbol → TradeLocker instrument ID."""
    if _client is None:
        return None
    if symbol in _instrument_cache:
        return _instrument_cache[symbol]
    try:
        iid = _client.get_instrument_id_from_symbol_name(symbol)
        _instrument_cache[symbol] = iid
        return iid
    except Exception:
        return None


def execute(symbol: str, action: str, lot: float,
            sl_price: float = None, tp_price: float = None):
    """
    Open a market order in TradeLocker mirroring the MT5 trade.

    Parameters
    ----------
    symbol : str
    action : 'buy' or 'sell'
    lot : float
    sl_price : float, optional
    tp_price : float, optional

    Returns
    -------
    order_id or None
    """
    if not TRADELOCKER_ENABLED or _client is None:
        return None

    try:
        iid = _get_instrument_id(symbol)
        if iid is None:
            print(f"❌ TradeLocker: símbolo {symbol} no encontrado")
            return None

        kwargs = {
            "quantity": float(lot),
            "side": action.lower(),
            "type_": "market",
        }
        if sl_price is not None:
            kwargs["stop_loss"] = float(sl_price)
            kwargs["stop_loss_type"] = "absolute"
        if tp_price is not None:
            kwargs["take_profit"] = float(tp_price)
            kwargs["take_profit_type"] = "absolute"

        order_id = _client.create_order(iid, **kwargs)
        print(f"✅ TradeLocker {symbol} {action} id={order_id}")
        return order_id
    except Exception as e:
        print(f"❌ TradeLocker error: {e}")
        return None
