# worker.py
"""
Resilient worker that processes a chunk of NASDAQ tickers.
- Ensures nasdaqlisted.txt exists (calls download_nasdaq_list.py if needed).
- Parses tickers robustly.
- Fetches ~14 months OHLCV via yfinance with retries.
- Computes indicators and writes raw-results-{JOB_INDEX}.json (always).
- Lots of logging to help diagnose "no tickers" problems.
"""

import os
import sys
import json
import math
import time
import traceback
import subprocess
from datetime import datetime
from pathlib import Path
import re

# Defensive imports
try:
    import pandas as pd
    import numpy as np
    import yfinance as yf
    # 'ta' library indicator classes
    from ta.trend import ADXIndicator
    from ta.volatility import AverageTrueRange
    from ta.volume import OnBalanceVolumeIndicator
    import requests
except Exception as e:
    print("[worker] IMPORT ERROR:", e)
    traceback.print_exc()
    # Attempt to write an artifact so aggregator isn't missing files
    JOB_INDEX = int(os.environ.get("JOB_INDEX", "0"))
    out = {"results": [], "attempted_count": 0, "processed_count": 0, "errors": [f"IMPORT ERROR: {str(e)}"], "job_index": JOB_INDEX, "ts": datetime.utcnow().isoformat()+"Z"}
    Path(".").mkdir(parents=True, exist_ok=True)
    with open(f"raw-results-{JOB_INDEX}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    sys.exit(0)

# CONFIG (can be tuned via env)
VOL_SPIKE_MULT = float(os.environ.get("VOL_SPIKE_MULT", "1.5"))
RECENT_DAYS_CROSSOVER = int(os.environ.get("RECENT_DAYS_CROSSOVER", "8"))
HISTORY_DAYS_FOR_SLOPE = int(os.environ.get("HISTORY_DAYS_FOR_SLOPE", "14"))
HISTORY_CHECK = HISTORY_DAYS_FOR_SLOPE + 1

JOB_TOTAL = int(os.environ.get("JOB_TOTAL", "1"))
JOB_INDEX = int(os.environ.get("JOB_INDEX", "0"))

OUT_FN = f"raw-results-{JOB_INDEX}.json"
NASDAQLIST = "nasdaqlisted.txt"

SYMBOL_RE = re.compile(r'^[A-Z0-9\.\-]{1,10}$')  # reasonable symbol pattern

# --- ensure nasdaqlisted.txt exists ---
def ensure_nasdaq_list():
    if Path(NASDAQLIST).exists():
        print(f"[worker] Found {NASDAQLIST} locally.")
        return True

    print("[worker] nasdaqlisted.txt missing — trying to run download_nasdaq_list.py")
    if Path("download_nasdaq_list.py").exists():
        try:
            res = subprocess.run([sys.executable, "download_nasdaq_list.py"], capture_output=True, text=True, timeout=120)
            print("[worker] download_nasdaq_list.py stdout:")
            print(res.stdout[:2000])
            if res.stderr:
                print("[worker] download_nasdaq_list.py stderr:")
                print(res.stderr[:2000])
        except Exception as e:
            print("[worker] Running download script failed:", e)
            traceback.print_exc()

    if Path(NASDAQLIST).exists():
        print("[worker] nasdaqlisted.txt now present after running download script.")
        return True

    # HTTP fallback direct
    try:
        url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        print(f"[worker] Attempting direct HTTP fetch of NASDAQ list from {url}")
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and r.content:
            Path(NASDAQLIST).write_bytes(r.content)
            print("[worker] Wrote nasdaqlisted.txt via HTTP fallback.")
            return True
        else:
            print(f"[worker] HTTP fallback returned status {r.status_code}")
    except Exception as e:
        print("[worker] HTTP fallback exception:", e)
        traceback.print_exc()

    return Path(NASDAQLIST).exists()

# --- robust ticker loader ---
def load_tickers(filename=NASDAQLIST):
    tickers = []
    try:
        with open(filename, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                # skip header / footer / meta lines
                low=line.lower()
                if low.startswith("symbol") or "file creation" in low or "nasdaqlisted" in low or line.startswith("#"):
                    continue
                parts = line.split("|")
                if not parts:
                    continue
                sym = parts[0].strip().upper()
                # Filter by regex and sanity
                if SYMBOL_RE.match(sym):
                    tickers.append(sym)
    except Exception as e:
        print("[worker] load_tickers error:", e)
        traceback.print_exc()
    # dedupe and sort for deterministic distribution
    tickers = sorted(list(dict.fromkeys(tickers)))
    return tickers

# --- chunking helpers ---
def chunk_list_round_robin(lst, total, index):
    # stable modulo-style distribution to avoid empty chunks when ceil chunking creates imbalance
    if total <= 1:
        return lst[:]
    return [t for i,t in enumerate(lst) if (i % total) == index]

def chunk_list_block(lst, total, index):
    # block chunking used previously (ceil). Keep as fallback.
    if total <= 1:
        return lst[:]
    n = len(lst)
    chunk_size = math.ceil(n / total)
    start = index * chunk_size
    end = min(start + chunk_size, n)
    return lst[start:end]

# --- small safe float helper ---
def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

# --- yfinance fetch with retries (exponential backoff) ---
def fetch_with_retries(ticker, retries=3, pause_base=1.0):
    last_exc = None
    for attempt in range(1, retries+1):
        try:
            df = yf.download(ticker, period="14mo", interval="1d", progress=False, threads=False)
            if df is None or df.empty:
                raise ValueError("empty df")
            return df
        except Exception as e:
            last_exc = e
            print(f"[fetch] {ticker} attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(pause_base * (2 ** (attempt-1)))
    # final attempt failed
    raise last_exc

# --- indicator computations (kept similar to original, using 'ta' for ADX/ATR/OBV) ---
def compute_indicators(df):
    try:
        if df is None or df.empty:
            return None
        df = df.copy()
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
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal
        out['macd_hist'] = safe_float(macd_hist.iloc[-1], 0.0)
        out['macd_slope'] = safe_float((macd_hist.iloc[-1] - macd_hist.iloc[-(HISTORY_DAYS_FOR_SLOPE+1)]) / HISTORY_DAYS_FOR_SLOPE, 0.0) if len(macd_hist) >= HISTORY_DAYS_FOR_SLOPE+1 else 0.0
        m_recent = macd_line.tail(RECENT_DAYS_CROSSOVER)
        s_recent = signal.tail(RECENT_DAYS_CROSSOVER)
        out['macd_bull'] = int((m_recent > s_recent).any() and (m_recent.iloc[-1] > s_recent.iloc[-1]) if len(m_recent)>0 and len(s_recent)>0 else False)

        # RSI 14
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(window=14).mean()
        roll_down = down.rolling(window=14).mean()
        rs = roll_up / (roll_down + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        out['rsi'] = safe_float(rsi.iloc[-1], 50.0)
        out['rsi_slope'] = safe_float((rsi.iloc[-1] - rsi.iloc[-(HISTORY_DAYS_FOR_SLOPE+1)]) / HISTORY_DAYS_FOR_SLOPE, 0.0) if len(rsi) >= HISTORY_DAYS_FOR_SLOPE+1 else 0.0

        # SMA20 & EMA50
        sma20 = close.rolling(window=20).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        out['sma20'] = safe_float(sma20.iloc[-1], safe_float(close.iloc[-1], 0.0))
        out['ema50'] = safe_float(ema50.iloc[-1], safe_float(close.iloc[-1], 0.0))
        out['above_trend'] = int((close.iloc[-1] > out['sma20']) and (close.iloc[-1] > out['ema50']))

        # Bollinger upper breakout
        std20 = close.rolling(window=20).std()
        bb_upper = sma20 + 2 * std20
        out['bb_breakout'] = int(close.iloc[-1] > safe_float(bb_upper.iloc[-1], 1e12))

        # ADX
        try:
            adx_ind = ADXIndicator(high=high, low=low, close=close, window=14)
            out['adx'] = safe_float(adx_ind.adx().iloc[-1], 0.0)
        except Exception:
            out['adx'] = 0.0

        # ATR
        try:
            atr_ind = AverageTrueRange(high=high, low=low, close=close, window=14)
            out['atr'] = safe_float(atr_ind.average_true_range().iloc[-1], 0.0)
        except Exception:
            out['atr'] = safe_float((high - low).rolling(14).mean().iloc[-1], 0.0)

        # OBV slope
        try:
            obv_ind = OnBalanceVolumeIndicator(close=close, volume=vol)
            obv = obv_ind.on_balance_volume()
            out['obv_slope'] = safe_float((obv.iloc[-1] - obv.iloc[-(HISTORY_DAYS_FOR_SLOPE+1)]) / HISTORY_DAYS_FOR_SLOPE, 0.0) if len(obv) >= HISTORY_DAYS_FOR_SLOPE+1 else 0.0
        except Exception:
            out['obv_slope'] = 0.0

        # Volume metrics
        vol20 = vol.rolling(window=20).mean()
        out['avg_vol20'] = safe_float(vol20.iloc[-1], safe_float(vol.iloc[-1], 0.0))
        out['vol_spike'] = int(safe_float(vol.iloc[-1], 0.0) > VOL_SPIKE_MULT * out['avg_vol20'])

        # Simple wave_strength: last peak vs sma20
        try:
            recent_close = close.tail(60)
            peaks = recent_close[(recent_close.shift(1) < recent_close) & (recent_close.shift(-1) < recent_close)]
            if len(peaks) > 0 and out['sma20'] > 0:
                out['wave_strength'] = safe_float(peaks.iloc[-1] / out['sma20'], 1.0)
            else:
                out['wave_strength'] = 1.0
        except Exception:
            out['wave_strength'] = 1.0

        out['last_close'] = safe_float(close.iloc[-1], 0.0)
        return out
    except Exception:
        print("[compute_indicators] EXCEPTION:")
        traceback.print_exc()
        return None

# --- main worker flow ---
def main():
    errors = []
    results = []
    attempted = 0
    processed = 0

    print(f"[worker] Starting job index={JOB_INDEX} total={JOB_TOTAL}")

    # ensure nasdaq list is present
    ok = ensure_nasdaq_list()
    if not ok:
        errors.append("nasdaqlisted.txt missing and download fallback failed.")
        print("[worker] ERROR: nasdaqlisted.txt missing. Aborting processing but will write artifact.")
        # write artifact and exit gracefully
        out = {"results": [], "attempted_count": 0, "processed_count": 0, "errors": errors, "job_index": JOB_INDEX, "ts": datetime.utcnow().isoformat()+"Z"}
        Path(".").mkdir(parents=True, exist_ok=True)
        with open(OUT_FN, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        sys.exit(0)

    # show first lines and counts for debugging
    try:
        with open(NASDAQLIST, "r", encoding="utf-8", errors="ignore") as fh:
            lines = [next(fh).rstrip("\n") for _ in range(10)]
        print("[worker] sample nasdaqlisted.txt first 10 lines:")
        for l in lines:
            print("  ", l)
    except Exception:
        print("[worker] could not read sample lines of nasdaqlisted.txt (maybe short file)")

    # load tickers robustly
    tickers = load_tickers()
    print(f"[worker] Parsed {len(tickers)} tickers from {NASDAQLIST}")

    # if no tickers, write artifact and exit (this is the "no tickers" case)
    if not tickers:
        errors.append("Parsed 0 tickers from nasdaqlisted.txt")
        out = {"results": [], "attempted_count": 0, "processed_count": 0, "errors": errors, "job_index": JOB_INDEX, "ts": datetime.utcnow().isoformat()+"Z"}
        Path(".").mkdir(parents=True, exist_ok=True)
        with open(OUT_FN, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print("[worker] Wrote empty artifact because no tickers parsed. Check nasdaqlisted.txt contents.")
        sys.exit(0)

    # chunk assignment: prefer round-robin (robust) but fall back to block chunking if needed
    assigned = chunk_list_round_robin(tickers, JOB_TOTAL, JOB_INDEX)
    if not assigned:
        assigned = chunk_list_block(tickers, JOB_TOTAL, JOB_INDEX)
    print(f"[worker] Assigned {len(assigned)} tickers to this job. Sample: {assigned[:10]}")

    attempted = len(assigned)
    for i, ticker in enumerate(assigned, 1):
        try:
            print(f"[worker] ({i}/{attempted}) Fetching {ticker}")
            df = fetch_with_retries(ticker, retries=3)
            # ensure needed columns
            df = df[['Open','High','Low','Close','Volume']].dropna()
            if df is None or df.empty or df.shape[0] < 30:
                print(f"[worker] ({i}/{attempted}) {ticker} - insufficient data, skipping")
                continue
            indicators = compute_indicators(df)
            if indicators is None:
                print(f"[worker] ({i}/{attempted}) {ticker} - indicators None, skipping")
                continue
            indicators['ticker'] = ticker
            indicators['ts'] = datetime.utcnow().isoformat() + "Z"
            results.append(indicators)
            processed += 1
            # slight pause to be polite to Yahoo endpoints
            time.sleep(0.08)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[worker] Exception processing {ticker}: {e}\n{tb}")
            errors.append(f"{ticker}: {str(e)}")
            continue

    # always write artifact
    out = {"results": results, "attempted_count": attempted, "processed_count": processed, "errors": errors, "job_index": JOB_INDEX, "ts": datetime.utcnow().isoformat()+"Z"}
    try:
        Path(".").mkdir(parents=True, exist_ok=True)
        with open(OUT_FN, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print(f"[worker] WROTE {OUT_FN} attempted={attempted} processed={processed} errors={len(errors)}")
    except Exception as e:
        print("[worker] Failed writing artifact:", e)
        traceback.print_exc()

    print("[worker] Completed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
