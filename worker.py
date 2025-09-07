# worker.py
"""
Hardened worker script to process a chunk of NASDAQ tickers.
- Uses JOB_TOTAL and JOB_INDEX env vars to determine assigned chunk.
- Writes raw-results-{JOB_INDEX}.json containing a list of per-ticker indicator dicts.
- Robust against missing nasdaqlisted.txt (attempts to run download_nasdaq_list.py).
- Fails loudly on import errors so GitHub Actions logs show the real cause.
"""

import os
import sys
import json
import math
import traceback
import subprocess
from datetime import datetime
from pathlib import Path

# Defensive imports: fail fast with a clear message if something missing
try:
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import pandas_ta as ta
    import requests
except Exception as e:
    print("[worker] ERROR importing required packages:", str(e))
    traceback.print_exc()
    sys.exit(3)

# Configuration
VOL_SPIKE_MULT = float(os.environ.get("VOL_SPIKE_MULT", "1.5"))
RECENT_DAYS_CROSSOVER = int(os.environ.get("RECENT_DAYS_CROSSOVER", "8"))
HISTORY_DAYS_FOR_SLOPE = int(os.environ.get("HISTORY_DAYS_FOR_SLOPE", "14"))
HISTORY_CHECK = HISTORY_DAYS_FOR_SLOPE + 1

JOB_TOTAL = int(os.environ.get("JOB_TOTAL", "1"))
JOB_INDEX = int(os.environ.get("JOB_INDEX", "0"))

OUT_FN_TEMPLATE = "raw-results-{}.json"

NASDAQLIST = "nasdaqlisted.txt"

def ensure_nasdaq_list():
    """Ensure nasdaqlisted.txt exists. If not, try running download_nasdaq_list.py.
       If that fails, try a direct HTTP fetch as a last resort."""
    if Path(NASDAQLIST).exists():
        print(f"[worker] Found {NASDAQLIST} locally.")
        return True

    print(f"[worker] {NASDAQLIST} not found — attempting to run download_nasdaq_list.py")
    # Try to execute the download script if present
    if Path("download_nasdaq_list.py").exists():
        try:
            res = subprocess.run([sys.executable, "download_nasdaq_list.py"], capture_output=True, text=True, timeout=120)
            print("[worker] download_nasdaq_list.py stdout:")
            print(res.stdout)
            if res.stderr:
                print("[worker] download_nasdaq_list.py stderr:")
                print(res.stderr)
        except subprocess.TimeoutExpired:
            print("[worker] download_nasdaq_list.py timed out")
        except Exception as e:
            print("[worker] Running download_nasdaq_list.py failed:", e)
            traceback.print_exc()

    # Re-check
    if Path(NASDAQLIST).exists():
        print(f"[worker] Successfully obtained {NASDAQLIST} via download script.")
        return True

    # Last-resort: attempt an HTTP download (lightweight, best-effort)
    try:
        url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        print(f"[worker] Attempting direct HTTP fetch of NASDAQ list from {url}")
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and r.content:
            Path(NASDAQLIST).write_bytes(r.content)
            print(f"[worker] Wrote {NASDAQLIST} via HTTP fallback.")
            return True
        else:
            print(f"[worker] HTTP fallback failed: status {r.status_code}")
    except Exception as e:
        print("[worker] HTTP fallback exception:", e)
        traceback.print_exc()

    print("[worker] FAILED to obtain nasdaqlisted.txt. Please ensure download_nasdaq_list.py exists and can run.")
    return False

def load_tickers(filename=NASDAQLIST):
    tickers = []
    try:
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
    except Exception as e:
        print(f"[worker] Error reading {filename}: {e}")
        return []

    # nasdaqlisted.txt has a header line; entries are pipe-separated, symbol is usually first column
    for line in lines:
        if not line or line.strip().startswith("NASDAQ") or line.lower().startswith("symbol|"):
            continue
        parts = line.split('|')
        if len(parts) < 1:
            continue
        sym = parts[0].strip()
        # skip footer lines like "File Creation Time:" or other meta lines
        if sym == "" or any(s in sym.lower() for s in ["file", "creation", "nasdaqlisted"]):
            continue
        tickers.append(sym)
    # dedupe/sort
    tickers = sorted(list(dict.fromkeys(tickers)))
    return tickers

def chunk_list(lst, total, index):
    if total <= 1:
        return lst[:]
    n = len(lst)
    chunk_size = math.ceil(n / total)
    start = index * chunk_size
    end = min(start + chunk_size, n)
    return lst[start:end]

def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, (int,float,np.floating)):
            return float(x)
        return float(np.nan_to_num(x, nan=default, posinf=default, neginf=default))
    except Exception:
        return default

def compute_indicators(df):
    """
    df: DataFrame with columns Open/High/Low/Close/Volume (or at least Close)
    Returns dict of indicators (or None if insufficient data)
    """
    try:
        if df is None or df.empty:
            return None
        df = df.copy()
        # Normalize column names
        df.columns = [c.capitalize() for c in df.columns]
        if 'Close' not in df.columns or 'High' not in df.columns or 'Low' not in df.columns:
            return None
        if df.shape[0] < HISTORY_CHECK:
            return None

        close = df['Close']
        high = df['High']
        low = df['Low']
        vol = df['Volume'] if 'Volume' in df.columns else pd.Series([0]*len(df), index=df.index)

        out = {}

        # MACD (fast 12, slow 26, signal 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal
        out['macd_hist'] = safe_float(macd_hist.iloc[-1], 0.0)
        # macd slope (14-day)
        if len(macd_hist) >= HISTORY_DAYS_FOR_SLOPE + 1:
            out['macd_slope'] = safe_float((macd_hist.iloc[-1] - macd_hist.iloc[-(HISTORY_DAYS_FOR_SLOPE+1)]) / HISTORY_DAYS_FOR_SLOPE, 0.0)
        else:
            out['macd_slope'] = 0.0

        # recent bullish crossover in last RECENT_DAYS_CROSSOVER days
        m_recent = macd_line.tail(RECENT_DAYS_CROSSOVER)
        s_recent = signal.tail(RECENT_DAYS_CROSSOVER)
        macd_bull = False
        try:
            if len(m_recent) > 0 and len(s_recent) > 0:
                if (m_recent > s_recent).any() and (m_recent.iloc[-1] > s_recent.iloc[-1]):
                    macd_bull = True
        except Exception:
            macd_bull = False
        out['macd_bull'] = int(macd_bull)

        # RSI(14)
        try:
            delta = close.diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            roll_up = up.rolling(window=14).mean()
            roll_down = down.rolling(window=14).mean()
            rs = roll_up / (roll_down + 1e-9)
            rsi = 100 - (100 / (1 + rs))
            out['rsi'] = safe_float(rsi.iloc[-1], 50.0)
            if len(rsi) >= HISTORY_DAYS_FOR_SLOPE + 1 and not pd.isna(rsi.iloc[-(HISTORY_DAYS_FOR_SLOPE+1)]):
                out['rsi_slope'] = safe_float((rsi.iloc[-1] - rsi.iloc[-(HISTORY_DAYS_FOR_SLOPE+1)]) / HISTORY_DAYS_FOR_SLOPE, 0.0)
            else:
                out['rsi_slope'] = 0.0
        except Exception:
            out['rsi'] = 50.0
            out['rsi_slope'] = 0.0

        # SMA20 & EMA50
        sma20 = close.rolling(window=20).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        out['sma20'] = safe_float(sma20.iloc[-1], safe_float(close.iloc[-1], 0.0))
        out['ema50'] = safe_float(ema50.iloc[-1], safe_float(close.iloc[-1], 0.0))
        out['above_trend'] = int((close.iloc[-1] > out['sma20']) and (close.iloc[-1] > out['ema50']))

        # Bollinger Bands (20,2)
        std20 = close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        out['bb_breakout'] = int(close.iloc[-1] > safe_float(bb_upper.iloc[-1], 1e12))

        # ADX (pandas_ta)
        try:
            adx_df = ta.adx(high=high, low=low, close=close, length=14)
            # find ADX column
            adx_col = next((c for c in adx_df.columns if str(c).upper().startswith("ADX")), None)
            if adx_col is not None:
                out['adx'] = safe_float(adx_df[adx_col].iloc[-1], 0.0)
            else:
                out['adx'] = 0.0
        except Exception:
            out['adx'] = 0.0

        # ATR (pandas_ta)
        try:
            atr = ta.atr(high=high, low=low, close=close, length=14)
            out['atr'] = safe_float(atr.iloc[-1], 0.0)
        except Exception:
            # fallback simple ATR-like estimate
            out['atr'] = safe_float((high - low).rolling(14).mean().iloc[-1], 0.0)

        # OBV slope
        try:
            obv = ta.obv(close=close, volume=vol)
            if isinstance(obv, pd.Series) or (hasattr(obv, 'iloc') and 'iloc' in dir(obv)):
                if len(obv) >= HISTORY_DAYS_FOR_SLOPE+1:
                    out['obv_slope'] = safe_float((obv.iloc[-1] - obv.iloc[-(HISTORY_DAYS_FOR_SLOPE+1)]) / HISTORY_DAYS_FOR_SLOPE, 0.0)
                else:
                    out['obv_slope'] = 0.0
            else:
                out['obv_slope'] = 0.0
        except Exception:
            out['obv_slope'] = 0.0

        # Volume metrics
        vol20 = vol.rolling(window=20).mean()
        out['avg_vol20'] = safe_float(vol20.iloc[-1], safe_float(vol.iloc[-1], 0.0))
        out['vol_spike'] = int(safe_float(vol.iloc[-1], 0.0) > VOL_SPIKE_MULT * out['avg_vol20'])

        # Simple Elliott-style wave_strength heuristic: last peak vs sma20
        try:
            recent_close = close.tail(60)
            peaks = recent_close[(recent_close.shift(1) < recent_close) & (recent_close.shift(-1) < recent_close)]
            if len(peaks) > 0 and not pd.isna(out['sma20']):
                last_peak = peaks.iloc[-1]
                out['wave_strength'] = safe_float(last_peak / out['sma20'], 1.0)
            else:
                out['wave_strength'] = 1.0
        except Exception:
            out['wave_strength'] = 1.0

        out['last_close'] = safe_float(close.iloc[-1], 0.0)
        return out
    except Exception as e:
        print("[compute_indicators] Unexpected exception:", e)
        traceback.print_exc()
        return None

def main():
    # Ensure nasdaq list present
    if not ensure_nasdaq_list():
        print("[worker] FATAL: cannot proceed without nasdaqlisted.txt")
        sys.exit(4)

    tickers = load_tickers(NASDAQLIST)
    if not tickers:
        print("[worker] No tickers loaded. Exiting.")
        sys.exit(5)

    assigned = chunk_list(tickers, JOB_TOTAL, JOB_INDEX)
    print(f"[worker] JOB_INDEX={JOB_INDEX} JOB_TOTAL={JOB_TOTAL} assigned {len(assigned)} tickers.")

    results = []
    total = len(assigned)
    for i, ticker in enumerate(assigned, 1):
        try:
            # Defensive fetch: threads=False and timeout via yfinance configuration not available,
            # but yfinance generally raises no exception; we guard with try/except anyway
            print(f"[worker] ({i}/{total}) {ticker} - fetching data")
            df = yf.download(ticker, period="14mo", interval="1d", progress=False, threads=False)
            if df is None or df.empty or df.shape[0] < 30:
                print(f"[worker] ({i}/{total}) {ticker} - insufficient data, skipping")
                continue
            # keep only necessary columns and drop rows with NA
            df = df[['Open','High','Low','Close','Volume']].dropna()
            indicators = compute_indicators(df)
            if indicators is None:
                print(f"[worker] ({i}/{total}) {ticker} - indicators None, skipping")
                continue
            indicators['ticker'] = ticker
            indicators['ts'] = datetime.utcnow().isoformat() + "Z"
            results.append(indicators)
        except Exception as e:
            print(f"[worker] Exception processing {ticker}: {e}")
            traceback.print_exc()
            # continue to next ticker
            continue

    out_fn = OUT_FN_TEMPLATE.format(JOB_INDEX)
    try:
        Path('.').mkdir(parents=True, exist_ok=True)
        with open(out_fn, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print(f"[worker] Wrote {out_fn} ({len(results)} tickers).")
    except Exception as e:
        print("[worker] Failed to write output file:", e)
        traceback.print_exc()
        sys.exit(6)

    # success
    print("[worker] Completed successfully.")
    sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[worker] Unhandled exception in __main__:", e)
        traceback.print_exc()
        sys.exit(7)
