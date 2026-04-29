import os
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta, time as dtime, timezone
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from joblib import load

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def env_required(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise RuntimeError(f"Falta variable de entorno requerida: {name}")
    return value


# ===================== CREDENCIALES =====================
# Seguridad: requeridas por entorno (sin secretos hardcodeados)
LOGIN = int(env_required("MT5_LOGIN"))
PASSWORD = env_required("MT5_PASSWORD")
SERVER = env_required("MT5_SERVER")

PAIRS = {
    "EURUSD": {
        "model": "models_eurusd/xgboost_model.joblib",
        "scaler": "models_eurusd/scaler.joblib",
        "features": ["RSI_14", "ATR_14", "MACD", "Spread", "ADX_14"],
        "min_pips": 10,
    },
    "GBPUSD": {
        "model": "models_gbpusd/xgboost_model.joblib",
        "scaler": "models_gbpusd/scaler.joblib",
        "features": ["RSI_14", "ATR_14", "MACD", "Spread", "ADX_14"],
        "min_pips": 12,
    },
}

LOT = float(os.getenv("BOT_LOT", "0.2"))
RETRIES = 5
LOOP_SECONDS = 60

CONF_THRESHOLD_BASE = 0.65
CONF_THRESHOLD_BASE_ALIGNED = 0.60
CONF_THRESHOLD_DEF = 0.72
CONF_THRESHOLD_DEF_ALIGNED = 0.68
TREND_BIAS = 0.10
DEVIATION = 10
MAGIC = 234567
ORDER_FILLING_MODE = os.getenv("ORDER_FILLING_MODE", "FOK").upper()
if ORDER_FILLING_MODE not in {"FOK", "IOC"}:
    raise RuntimeError("ORDER_FILLING_MODE inválido. Usa FOK o IOC.")

BUFFER_PIPS = 30
TREND_EMA = 200
TREND_LOOKBACK = 220
STRUCT_WINDOW = 20
NEED_BARS = 500

NY_TZ = ZoneInfo("America/New_York")
LON_TZ = ZoneInfo("Europe/London")
TRADE_TIMEFRAME = mt5.TIMEFRAME_M15
VWAP_TIMEFRAME = mt5.TIMEFRAME_M15

VWAP_TOLERANCE_PIPS = 5
LEVEL_TOLERANCE_PIPS = 5

VP_DAYS = 15
VP_BIN_PIPS = 1.0
VP_MIN_REPEAT_DAYS = 1
VP_HVN_TOP = 3

DISABLED_PAIRS = set()
LAST_CLOSE_UTC_PER_SYMBOL = {}
LAST_OPEN_TICKET = {}
LAST_OPEN_DIR = {}
LAST_TP_PRICE = {}
LAST_TP_DIR = {}

COOLDOWN_SECONDS = 30 * 60

# Trailing stop
TRAIL_ACTIVATE_MULT = 1.0
TRAIL_ATR_MULT = 1.0
TRAIL_ATR_MULT_PHASE2 = 3.0
TRAIL_PHASE2_TRIGGER = 2.0

# -------- Horarios --------
LONDON_START = dtime(2, 0)
LONDON_END = dtime(12, 0)
NY_START = dtime(2, 0)
NY_END = dtime(12, 0)

# Bloqueo de noticias (UTC): "2026-05-01T12:30:00+00:00,2026-05-02T14:00:00+00:00"
NEWS_EVENTS_UTC = [
    datetime.fromisoformat(x.strip())
    for x in os.getenv("NEWS_EVENTS_UTC", "").split(",")
    if x.strip()
]
NEWS_BLOCK_MINUTES = int(os.getenv("NEWS_BLOCK_MINUTES", "30"))
INVALID_FILL_RETCODE = getattr(mt5, "TRADE_RETCODE_INVALID_FILL", None)
INVALID_GENERIC_RETCODE = getattr(mt5, "TRADE_RETCODE_INVALID", None)
MODEL_CACHE = {}
SCALER_CACHE = {}
USE_HEURISTIC_FALLBACK = os.getenv("USE_HEURISTIC_FALLBACK", "1").strip() not in {"0", "false", "False"}


def is_news_block(now_utc: datetime) -> bool:
    for event_dt in NEWS_EVENTS_UTC:
        if abs((event_dt - now_utc).total_seconds()) <= NEWS_BLOCK_MINUTES * 60:
            return True
    return False


def is_trading_time(now_utc: datetime) -> bool:
    now_lon_dt = now_utc.astimezone(LON_TZ)
    if now_lon_dt.weekday() in (5, 6):
        return False

    now_lon = now_lon_dt.time()
    now_ny = now_utc.astimezone(NY_TZ).time()
    in_london = LONDON_START <= now_lon <= LONDON_END
    in_ny = NY_START <= now_ny <= NY_END
    return in_london or in_ny


def pip_size(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    if info is None:
        return 0.0001
    return info.point * 10 if info.digits in (3, 5) else info.point


def ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period).mean()


def macd(series, fast=12, slow=26, sig=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal = ema(macd_line, sig)
    hist = macd_line - signal
    return macd_line, signal, hist


def adx(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_tr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).sum() / atr_tr)
    minus_di = 100 * (minus_dm.rolling(window=period).sum() / atr_tr)
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8))
    return dx.rolling(window=period).mean()


def rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / period, min_periods=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def agregar_indicadores(df: pd.DataFrame):
    df["RSI_14"] = rsi(df["close"], 14)
    df["ATR_14"] = atr(df, 14)
    macd_line, _, _ = macd(df["close"])
    df["MACD"] = macd_line
    df["ADX_14"] = adx(df, 14)
    df["Spread"] = df["high"] - df["low"]
    return df.dropna().reset_index(drop=True)


def detect_regime(df: pd.DataFrame) -> str:
    if len(df) < 120:
        return "unknown"
    adx_now = float(df["ADX_14"].iloc[-1])
    atr_now = float(df["ATR_14"].iloc[-1])
    atr_med = float(df["ATR_14"].tail(100).median())

    trend_state = "trend" if adx_now >= 22 else "range"
    vol_state = "high_vol" if atr_now > atr_med * 1.25 else "normal_vol"
    return f"{trend_state}_{vol_state}"


def trend_filter_closed(df):
    df_closed = df.iloc[:-1]
    if len(df_closed) < TREND_LOOKBACK:
        return None, None
    closes = df_closed["close"]
    ema_series = closes.ewm(span=TREND_EMA, adjust=False).mean()
    ema_val = float(ema_series.iloc[-1])
    last_candle = df_closed.iloc[-1]
    if last_candle["low"] > ema_val:
        return "bull", ema_val
    if last_candle["high"] < ema_val:
        return "bear", ema_val
    return None, ema_val


def connect_mt5():
    for i in range(RETRIES):
        if mt5.initialize(login=LOGIN, server=SERVER, password=PASSWORD):
            print("Conectado a MT5")
            return True
        else:
            print(f"Falló conexión ({i+1}/{RETRIES}): {mt5.last_error()}")
            time.sleep(2)
    return False


def ensure_symbol(symbol):
    return mt5.symbol_select(symbol, True)


def get_vwap_range(symbol, start_utc, end_utc):
    rates = mt5.copy_rates_range(symbol, VWAP_TIMEFRAME, start_utc, end_utc)
    if rates is None or len(rates) < 5:
        return None
    df = pd.DataFrame(rates)
    vol = df["tick_volume"] if "tick_volume" in df.columns else df["volume"]
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * vol
    return float((pv.cumsum() / vol.cumsum()).iloc[-1])


def get_daily_vwap(symbol):
    now_utc = datetime.now(timezone.utc)
    now_ny = now_utc.astimezone(NY_TZ)
    start_ny = datetime(now_ny.year, now_ny.month, now_ny.day, tzinfo=NY_TZ)
    return get_vwap_range(symbol, start_ny.astimezone(timezone.utc), now_utc)


def get_weekly_vwap(symbol):
    now_utc = datetime.now(timezone.utc)
    now_ny = now_utc.astimezone(NY_TZ)
    start_week = now_ny - timedelta(days=now_ny.weekday())
    start_ny = datetime(start_week.year, start_week.month, start_week.day, tzinfo=NY_TZ)
    return get_vwap_range(symbol, start_ny.astimezone(timezone.utc), now_utc)


def get_monthly_vwap(symbol):
    now_utc = datetime.now(timezone.utc)
    now_ny = now_utc.astimezone(NY_TZ)
    start_ny = datetime(now_ny.year, now_ny.month, 1, tzinfo=NY_TZ)
    return get_vwap_range(symbol, start_ny.astimezone(timezone.utc), now_utc)


def vwap_alignment_ok(action, price, vwap_d, vwap_w, vwap_m, pip):
    if vwap_d is None or vwap_w is None or vwap_m is None:
        return False
    tol = VWAP_TOLERANCE_PIPS * pip
    if action == "buy":
        return (vwap_d <= price + tol) and (vwap_w <= price + tol) and (vwap_m <= price + tol)
    return (vwap_d >= price - tol) and (vwap_w >= price - tol) and (vwap_m >= price - tol)


def build_volume_profile_levels(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        return []
    pip = pip_size(symbol)
    bin_size = VP_BIN_PIPS * pip
    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=VP_DAYS)

    rates = mt5.copy_rates_range(symbol, TRADE_TIMEFRAME, start_utc, end_utc)
    if rates is None or len(rates) < 50:
        return []

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time_ny"] = df["time"].dt.tz_convert(NY_TZ)
    df["day"] = df["time_ny"].dt.date

    levels = []
    for _, d in df.groupby("day"):
        if len(d) < 10:
            continue
        typical = (d["high"] + d["low"] + d["close"]) / 3
        vol = d["tick_volume"] if "tick_volume" in d.columns else d["volume"]
        bins = (typical / bin_size).round().astype(int)
        vol_by_bin = defaultdict(float)
        for b, v in zip(bins, vol):
            vol_by_bin[b] += float(v)
        if not vol_by_bin:
            continue
        poc_bin = max(vol_by_bin, key=vol_by_bin.get)
        levels.append(poc_bin * bin_size)
        top_bins = sorted(vol_by_bin.items(), key=lambda x: x[1], reverse=True)[:VP_HVN_TOP]
        levels.extend(b * bin_size for b, _ in top_bins)
    return levels


def find_strong_levels(symbol):
    levels = build_volume_profile_levels(symbol)
    if not levels:
        return []
    pip = pip_size(symbol)
    bin_size = VP_BIN_PIPS * pip
    rounded = [(round(l / bin_size) * bin_size) for l in levels]
    counts = Counter(rounded)
    strong = {lvl for lvl, cnt in counts.items() if cnt >= VP_MIN_REPEAT_DAYS}
    if not strong:
        strong = set(rounded)
    return sorted(strong)


def nearest_level_distance_pips(symbol, price, levels):
    pip = pip_size(symbol)
    if not levels:
        return None, None
    nearest = min(levels, key=lambda lvl: abs(price - lvl))
    return nearest, abs(price - nearest) / pip


def can_trade_after_cooldown(symbol, now_utc):
    if symbol not in LAST_CLOSE_UTC_PER_SYMBOL:
        return True
    elapsed = (now_utc - LAST_CLOSE_UTC_PER_SYMBOL[symbol]).total_seconds()
    return elapsed >= COOLDOWN_SECONDS


def set_cooldown(symbol, now_utc):
    LAST_CLOSE_UTC_PER_SYMBOL[symbol] = now_utc


def heuristic_signal(df: pd.DataFrame, trend: str, sym: str):
    if len(df) < 3:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    rsi_now = float(last["RSI_14"])
    macd_now = float(last["MACD"])
    macd_prev = float(prev["MACD"])
    adx_now = float(last["ADX_14"])

    buy_score = 0.0
    sell_score = 0.0

    if trend == "bull":
        buy_score += 0.30
    elif trend == "bear":
        sell_score += 0.30

    if 35 <= rsi_now <= 65:
        buy_score += 0.10
        sell_score += 0.10
    elif rsi_now < 35:
        buy_score += 0.25
    elif rsi_now > 65:
        sell_score += 0.25

    if macd_now > macd_prev:
        buy_score += 0.25
    elif macd_now < macd_prev:
        sell_score += 0.25

    if adx_now >= 22:
        buy_score += 0.20
        sell_score += 0.20

    if macd_now > 0:
        buy_score += 0.15
    elif macd_now < 0:
        sell_score += 0.15

    total = buy_score + sell_score
    if total <= 0:
        return None

    p_sell = sell_score / total
    p_buy = buy_score / total
    best_class = 1 if p_buy >= p_sell else 0
    best_conf = max(p_buy, p_sell)
    print(
        f"{sym} → HEUR BUY: {p_buy*100:.2f}% | HEUR SELL: {p_sell*100:.2f}% | acción={'BUY' if best_class==1 else 'SELL'} | conf={best_conf:.4f}"
    )
    return best_class, best_conf, np.array([p_sell, p_buy], dtype=float)


def ml_signal_with_trend_bias(df, trend, model_path, scaler_path, feature_cols, sym):
    if sym in DISABLED_PAIRS:
        return None
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print(f"{sym}: falta modelo o scaler.")
        if USE_HEURISTIC_FALLBACK:
            return heuristic_signal(df, trend, sym)
        return None
    try:
        x_live = pd.DataFrame(df[feature_cols].astype(np.float64).values[[-1]], columns=feature_cols)
        if scaler_path not in SCALER_CACHE:
            SCALER_CACHE[scaler_path] = load(scaler_path)
        if model_path not in MODEL_CACHE:
            MODEL_CACHE[model_path] = load(model_path)
        scaler = SCALER_CACHE[scaler_path]
        model = MODEL_CACHE[model_path]
        if hasattr(scaler, "feature_names_in_"):
            expected = list(scaler.feature_names_in_)
            if expected != feature_cols:
                print(f"{sym}: feature mismatch.")
                if USE_HEURISTIC_FALLBACK:
                    return heuristic_signal(df, trend, sym)
                DISABLED_PAIRS.add(sym)
                return None
        x_scaled = scaler.transform(x_live)
    except Exception as e:
        print(f"{sym}: error ML {e}")
        if USE_HEURISTIC_FALLBACK:
            return heuristic_signal(df, trend, sym)
        DISABLED_PAIRS.add(sym)
        return None

    probas = model.predict_proba(x_scaled)[0].astype(float)
    if trend == "bull":
        probas[1] += TREND_BIAS
    elif trend == "bear":
        probas[0] += TREND_BIAS
    probas = np.clip(probas, 1e-9, None)
    probas = probas / probas.sum()

    best_class = int(np.argmax(probas))
    best_conf = float(probas[best_class])
    print(
        f"{sym} → ML BUY: {probas[1]*100:.2f}% | ML SELL: {probas[0]*100:.2f}% | acción={'BUY' if best_class==1 else 'SELL'} | conf={best_conf:.4f}"
    )
    return best_class, best_conf, probas


def regime_conf_threshold(regime: str, aligned: bool) -> float:
    if regime == "trend_normal_vol":
        return CONF_THRESHOLD_BASE_ALIGNED if aligned else CONF_THRESHOLD_BASE
    return CONF_THRESHOLD_DEF_ALIGNED if aligned else CONF_THRESHOLD_DEF


def calc_sl_tp_from_structure(df, action):
    recent = df.iloc[-STRUCT_WINDOW:]
    return float(recent["low"].min() if action == "buy" else recent["high"].max())


def calc_sl_tp_with_rr(symbol, action, df):
    pip = pip_size(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"{symbol}: sin tick.")
    if action == "buy":
        if tick.ask is None:
            raise RuntimeError(f"{symbol}: ask no disponible.")
        price = tick.ask
    else:
        if tick.bid is None:
            raise RuntimeError(f"{symbol}: bid no disponible.")
        price = tick.bid

    extreme = calc_sl_tp_from_structure(df, action)
    buffer = BUFFER_PIPS * pip
    if action == "buy":
        sl = extreme - buffer
        risk = price - sl
        tp = price + (risk * 2)
    else:
        sl = extreme + buffer
        risk = sl - price
        tp = price - (risk * 2)

    risk_pips = risk / pip
    return price, sl, tp, risk, risk_pips


def modify_position_sltp(symbol, ticket, sl, tp, magic, comment):
    req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol": symbol,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": magic,
        "comment": comment,
    }
    result = mt5.order_send(req)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        res_comment = None if result is None else getattr(result, "comment", None)
        print(
            f"{symbol}: error al actualizar {comment} ticket={ticket} ret={None if result is None else result.retcode} "
            f"res_comment={res_comment} last_error={mt5.last_error()}"
        )
    return result


def update_trailing_sl_dynamic(symbol, df):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return

    pip = pip_size(symbol)
    atr_val = float(df["ATR_14"].iloc[-1])
    trail_dist_phase1 = max(atr_val * TRAIL_ATR_MULT, 2 * pip)
    trail_dist_phase2 = max(atr_val * TRAIL_ATR_MULT_PHASE2, 2 * pip)
    activate_dist = atr_val * TRAIL_ACTIVATE_MULT
    phase2_trigger = atr_val * TRAIL_PHASE2_TRIGGER

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return

    for pos in positions:
        direction = "buy" if pos.type == mt5.POSITION_TYPE_BUY else "sell"
        price_open = pos.price_open
        current_sl = pos.sl
        tp = pos.tp

        if tp and tp != 0:
            if direction == "buy":
                tp_dist = tp - price_open
                if tp_dist > 0 and (tick.bid - price_open) >= (0.5 * tp_dist):
                    if current_sl == 0 or current_sl < price_open:
                        modify_position_sltp(symbol, pos.ticket, price_open, pos.tp, pos.magic, "BE-50")
            else:
                tp_dist = price_open - tp
                if tp_dist > 0 and (price_open - tick.ask) >= (0.5 * tp_dist):
                    if current_sl == 0 or current_sl > price_open:
                        modify_position_sltp(symbol, pos.ticket, price_open, pos.tp, pos.magic, "BE-50")

        if direction == "buy":
            move = tick.bid - price_open
            if move >= activate_dist and move < phase2_trigger:
                new_sl = tick.bid - trail_dist_phase1
                if current_sl == 0 or new_sl > current_sl + pip * 0.1:
                    modify_position_sltp(symbol, pos.ticket, new_sl, pos.tp, pos.magic, "TRAIL-P1")
            elif move >= phase2_trigger:
                new_sl = tick.bid - trail_dist_phase2
                if current_sl == 0 or new_sl > current_sl + pip * 0.1:
                    modify_position_sltp(symbol, pos.ticket, new_sl, pos.tp, pos.magic, "TRAIL-P2")
        else:
            move = price_open - tick.ask
            if move >= activate_dist and move < phase2_trigger:
                new_sl = tick.ask + trail_dist_phase1
                if current_sl == 0 or new_sl < current_sl - pip * 0.1:
                    modify_position_sltp(symbol, pos.ticket, new_sl, pos.tp, pos.magic, "TRAIL-P1")
            elif move >= phase2_trigger:
                new_sl = tick.ask + trail_dist_phase2
                if current_sl == 0 or new_sl < current_sl - pip * 0.1:
                    modify_position_sltp(symbol, pos.ticket, new_sl, pos.tp, pos.magic, "TRAIL-P2")


def send_order(symbol, action, lot, df):
    price, sl, tp, _, _ = calc_sl_tp_with_rr(symbol, action, df)
    order_type = mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL
    if ORDER_FILLING_MODE == "IOC":
        filling_type = mt5.ORDER_FILLING_IOC
    else:
        filling_type = mt5.ORDER_FILLING_FOK
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": MAGIC,
        "comment": "ML-BOT-FX",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_type,
    }
    result = mt5.order_send(req)
    fallback_retcodes = {
        code for code in (INVALID_FILL_RETCODE, INVALID_GENERIC_RETCODE) if code is not None
    }
    if result is not None and result.retcode in fallback_retcodes:
        req["type_filling"] = mt5.ORDER_FILLING_IOC
        result = mt5.order_send(req)
    return result


def get_last_close_info(symbol):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=2)
    deals = mt5.history_deals_get(start, end)
    if deals is None:
        return None
    sym_deals = [d for d in deals if d.symbol == symbol and d.entry == mt5.DEAL_ENTRY_OUT]
    if not sym_deals:
        return None
    return sorted(sym_deals, key=lambda d: d.time)[-1]


def main():
    if not connect_mt5():
        return

    for sym in PAIRS:
        if not ensure_symbol(sym):
            print(f"No se pudo seleccionar símbolo {sym}")
            mt5.shutdown()
            return

    while True:
        loop_start = time.time()
        now_utc = datetime.now(timezone.utc)
        print(f"⏳ Loop UTC: {now_utc.isoformat()}")

        if is_news_block(now_utc):
            print("Bloqueado por ventana de noticias macro.")
            elapsed = time.time() - loop_start
            time.sleep(max(0, LOOP_SECONDS - elapsed))
            continue

        if not is_trading_time(now_utc):
            print("Fuera de horario o viernes bloqueado.")
            elapsed = time.time() - loop_start
            time.sleep(max(0, LOOP_SECONDS - elapsed))
            continue

        for sym, cfg in PAIRS.items():
            print(f"--- {sym} ---")

            pos_list = mt5.positions_get(symbol=sym)
            if pos_list:
                LAST_OPEN_TICKET[sym] = pos_list[0].ticket
                rates_trail = mt5.copy_rates_from_pos(sym, TRADE_TIMEFRAME, 0, NEED_BARS)
                if rates_trail is not None and len(rates_trail) >= NEED_BARS:
                    df_trail = pd.DataFrame(rates_trail)
                    df_trail["time"] = pd.to_datetime(df_trail["time"], unit="s")
                    df_trail = agregar_indicadores(df_trail)
                    update_trailing_sl_dynamic(sym, df_trail)
                print(f"{sym}: ya hay posición abierta.")
                continue

            if sym in LAST_OPEN_TICKET:
                close_info = get_last_close_info(sym)
                if close_info is not None and sym in LAST_OPEN_DIR and close_info.profit > 0:
                    LAST_TP_DIR[sym] = LAST_OPEN_DIR[sym]
                    LAST_TP_PRICE[sym] = close_info.price
                set_cooldown(sym, datetime.now(timezone.utc))
                del LAST_OPEN_TICKET[sym]

            if not can_trade_after_cooldown(sym, now_utc):
                print(f"{sym}: en cooldown.")
                continue

            rates = mt5.copy_rates_from_pos(sym, TRADE_TIMEFRAME, 0, NEED_BARS)
            if rates is None or len(rates) < NEED_BARS:
                print(f"{sym}: sin datos suficientes")
                continue

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = agregar_indicadores(df)

            trend, ema_val = trend_filter_closed(df)
            print(f"{sym}: trend={trend} ema={ema_val}")
            if trend is None:
                continue

            tick = mt5.symbol_info_tick(sym)
            if tick is None:
                print(f"{sym}: sin tick")
                continue
            if tick.ask is not None:
                price = tick.ask
            elif tick.bid is not None:
                price = tick.bid
            else:
                print(f"{sym}: tick sin ask/bid")
                continue
            pip = pip_size(sym)

            vwap_d = get_daily_vwap(sym)
            vwap_w = get_weekly_vwap(sym)
            vwap_m = get_monthly_vwap(sym)
            strong_levels = find_strong_levels(sym)
            nearest_level, level_dist = nearest_level_distance_pips(sym, price, strong_levels)
            print(f"{sym}: level={nearest_level} dist={level_dist}")

            sig = ml_signal_with_trend_bias(df, trend, cfg["model"], cfg["scaler"], cfg["features"], sym)
            if sig is None:
                continue
            best_class, best_conf, _ = sig
            action = "buy" if best_class == 1 else "sell"

            aligned = (action == "buy" and trend == "bull") or (action == "sell" and trend == "bear")
            if not aligned:
                print(f"{sym}: EMA en contra.")
                continue

            regime = detect_regime(df)
            conf_threshold = regime_conf_threshold(regime, aligned=True)
            if best_conf < conf_threshold:
                print(f"{sym}: conf baja {best_conf:.4f} < {conf_threshold:.4f} (regime={regime})")
                continue

            if not vwap_alignment_ok(action, price, vwap_d, vwap_w, vwap_m, pip):
                print(f"{sym}: VWAP D/W/M no alineado.")
                continue

            if sym in LAST_TP_DIR and sym in LAST_TP_PRICE and LAST_TP_DIR[sym] == action:
                atr_val = float(df["ATR_14"].iloc[-1])
                allow_reentry = (
                    (action == "sell" and price > (LAST_TP_PRICE[sym] + atr_val))
                    or (action == "buy" and price < (LAST_TP_PRICE[sym] - atr_val))
                )
                if not allow_reentry:
                    print(f"{sym}: evita re-entrada mismo impulso.")
                    continue

            if level_dist is None or level_dist > LEVEL_TOLERANCE_PIPS:
                print(f"{sym}: lejos del nivel fuerte.")
                continue

            try:
                _, _, _, _, risk_pips = calc_sl_tp_with_rr(sym, action, df)
            except RuntimeError as e:
                print(str(e))
                continue

            if risk_pips < float(cfg["min_pips"]):
                print(f"{sym}: riesgo estructural {risk_pips:.2f} pips < min_pips {cfg['min_pips']}")
                continue

            r = send_order(sym, action, LOT, df)
            if r is not None and r.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"{sym}: ORDEN EJECUTADA {action.upper()}")
                LAST_OPEN_DIR[sym] = action
            else:
                print(f"{sym}: orden falló ret={None if r is None else r.retcode}")

        elapsed = time.time() - loop_start
        time.sleep(max(0, LOOP_SECONDS - elapsed))


if __name__ == "__main__":
    try:
        main()
    finally:
        mt5.shutdown()
