# worker.py
import os
import json
import math
from datetime import datetime
import traceback

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta

# Configurable thresholds
VOL_SPIKE_MULT = 1.5
RECENT_DAYS_CROSSOVER = 8
HISTORY_DAYS_FOR_SLOPE = 14
HISTORY_CHECK = HISTORY_DAYS_FOR_SLOPE + 1

# Environment variables to split work
JOB_TOTAL = int(os.environ.get("JOB_TOTAL", "1"))
JOB_INDEX = int(os.environ.get("JOB_INDEX", "0"))

def load_tickers(filename="nasdaqlisted.txt"):
    tickers = []
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    # skip header
    for line in lines[1:]:
        if not line.strip(): 
            continue
        parts = line.split('|')
        if len(parts) >= 2:
            sym = parts[0].strip()
            # skip test/invalid lines (file footer lines may be present)
            if sym.upper() == "FILE HEADER" or sym.startswith("Symbol"):
                continue
            # sometimes there's extra fields; the first column in nasdaqlisted.txt is the symbol
            tickers.append(sym)
    # remove duplicate & sort
    tickers = sorted(list(set(tickers)))
    return tickers

def chunk_list(lst, total, index):
    n = len(lst)
    chunk_size = math.ceil(n / total)
    start = index * chunk_size
    end = start + chunk_size
    return lst[start:end]

def compute_indicators(df):
    # ensure required columns
    df = df.copy()
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    # ensure we have enough rows
    out = {}
    if df.shape[0] < HISTORY_CHECK:
        return None

    close = df['Close']
    high = df['High']
    low = df['Low']
    vol = df['Volume']

    # MACD (using EMAs)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal

    out['macd_hist'] = float(macd_hist.iloc[-1])
    # macd 14-day slope
    if len(macd_hist) >= HISTORY_DAYS_FOR_SLOPE+1:
        out['macd_slope'] = float((macd_hist.iloc[-1] - macd_hist.iloc[-(HISTORY_DAYS_FOR_SLOPE+1)]) / HISTORY_DAYS_FOR_SLOPE)
    else:
        out['macd_slope'] = 0.0

    # recent bullish crossover check (within last RECENT_DAYS_CROSSOVER days)
    recent = macd_line.tail(RECENT_DAYS_CROSSOVER)
    recent_sig = signal.tail(RECENT_DAYS_CROSSOVER)
    macd_bull = False
    try:
        if (recent.iloc[-1] > recent_sig.iloc[-1]) and ((recent > recent_sig).any()):
            macd_bull = True
    except Exception:
        macd_bull = False
    out['macd_bull'] = int(macd_bull)

    # RSI (14)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    out['rsi'] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    # rsi slope
    if len(rsi) >= HISTORY_DAYS_FOR_SLOPE+1 and not pd.isna(rsi.iloc[-(HISTORY_DAYS_FOR_SLOPE+1)]):
        out['rsi_slope'] = float((rsi.iloc[-1] - rsi.iloc[-(HISTORY_DAYS_FOR_SLOPE+1)]) / HISTORY_DAYS_FOR_SLOPE)
    else:
        out['rsi_slope'] = 0.0

    # SMA20, EMA50
    out['sma20'] = float(close.rolling(window=20).mean().iloc[-1])
    out['ema50'] = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    out['above_trend'] = int((close.iloc[-1] > out['sma20']) and (close.iloc[-1] > out['ema50']))

    # Bollinger Bands (20, 2)
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    bb_upper = sma20 + 2 * std20
    out['bb_breakout'] = int(close.iloc[-1] > bb_upper.iloc[-1])

    # ADX (via pandas_ta)
    try:
        adx_df = ta.adx(high=high, low=low, close=close, length=14)
        # pandas_ta returns ADX_{length}
        adx_col = next((c for c in adx_df.columns if str(c).upper().startswith("ADX")), None)
        if adx_col is not None:
            out['adx'] = float(adx_df[adx_col].iloc[-1])
        else:
            out['adx'] = 0.0
    except Exception:
        out['adx'] = 0.0

    # ATR
    try:
        atr = ta.atr(high=high, low=low, close=close, length=14)
        out['atr'] = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
    except Exception:
        out['atr'] = float((high - low).rolling(14).mean().iloc[-1])

    # OBV slope (pandas_ta)
    try:
        obv = ta.obv(close=close, volume=vol)
        out['obv_slope'] = float((obv.iloc[-1] - obv.iloc[-(HISTORY_DAYS_FOR_SLOPE+1)]) / HISTORY_DAYS_FOR_SLOPE) if len(obv) >= HISTORY_DAYS_FOR_SLOPE+1 else 0.0
    except Exception:
        out['obv_slope'] = 0.0

    # Volume spike vs 20-day average
    vol20 = vol.rolling(window=20).mean()
    out['avg_vol20'] = float(vol20.iloc[-1]) if not pd.isna(vol20.iloc[-1]) else float(vol.iloc[-1])
    out['vol_spike'] = int(vol.iloc[-1] > VOL_SPIKE_MULT * out['avg_vol20'])

    # Wave strength: simple peak vs sma20 heuristic
    try:
        recent_close = close.tail(60)
        peaks = recent_close[(recent_close.shift(1) < recent_close) & (recent_close.shift(-1) < recent_close)]
        if len(peaks) > 0 and not pd.isna(out['sma20']):
            last_peak = peaks.iloc[-1]
            out['wave_strength'] = float(last_peak / out['sma20'])
        else:
            out['wave_strength'] = 1.0
    except Exception:
        out['wave_strength'] = 1.0

    # Keep a few extra values
    out['last_close'] = float(close.iloc[-1])
    out['ticker'] = None  # filled by caller
    return out

def main():
    tickers = load_tickers("nasdaqlisted.txt")
    assigned = chunk_list(tickers, JOB_TOTAL, JOB_INDEX)
    print(f"JOB {JOB_INDEX}/{JOB_TOTAL} assigned {len(assigned)} tickers")

    results = []
    for i, ticker in enumerate(assigned):
        try:
            # fetch 14 months (~420 trading days) of daily data
            df = yf.download(ticker, period="14mo", interval="1d", progress=False, threads=False)
            if df.empty or df.shape[0] < 30:
                # skip tiny popul
                continue
            df = df[['Open','High','Low','Close','Volume']].dropna()
            indicators = compute_indicators(df)
            if indicators is None:
                continue
            indicators['ticker'] = ticker
            # Add timestamp
            indicators['ts'] = datetime.utcnow().isoformat() + "Z"
            results.append(indicators)
        except Exception as e:
            # don't crash whole worker if a single ticker fails
            print(f"Error for {ticker}: {e}")
            print(traceback.format_exc())
            continue

    # Save raw results for aggregator
    out_fn = f"raw-results-{JOB_INDEX}.json"
    with open(out_fn, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_fn}, {len(results)} tickers processed.")

if __name__ == "__main__":
    main()
