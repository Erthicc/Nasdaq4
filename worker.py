# worker.py
"""
Resilient worker with yfinance + Stooq fallback.
- Tries yfinance first; if yfinance returns empty, falls back to Stooq CSV.
- Writes raw-results-{JOB_INDEX}.json always.
- Includes AAPL health check and detailed logging.
"""
import os
import sys
import json
import math
import time
import traceback
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import re
from io import StringIO

# Defensive imports
try:
    import pandas as pd
    import numpy as np
    import yfinance as yf
    from ta.trend import ADXIndicator
    from ta.volatility import AverageTrueRange
    from ta.volume import OnBalanceVolumeIndicator
    import requests
except Exception as e:
    print("[worker] IMPORT ERROR:", e)
    traceback.print_exc()
    JOB_INDEX = int(os.environ.get("JOB_INDEX", "0"))
    out = {"results": [], "attempted_count": 0, "processed_count": 0, "errors": [f"IMPORT ERROR: {str(e)}"], "job_index": JOB_INDEX, "ts": datetime.utcnow().isoformat()+"Z"}
    Path(".").mkdir(parents=True, exist_ok=True)
    with open(f"raw-results-{JOB_INDEX}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    sys.exit(0)

# Config (tunable)
VOL_SPIKE_MULT = float(os.environ.get("VOL_SPIKE_MULT", "1.5"))
RECENT_DAYS_CROSSOVER = int(os.environ.get("RECENT_DAYS_CROSSOVER", "8"))
HISTORY_DAYS_FOR_SLOPE = int(os.environ.get("HISTORY_DAYS_FOR_SLOPE", "14"))
HISTORY_CHECK = HISTORY_DAYS_FOR_SLOPE + 1

JOB_TOTAL = int(os.environ.get("JOB_TOTAL", "1"))
JOB_INDEX = int(os.environ.get("JOB_INDEX", "0"))

OUT_FN = f"raw-results-{JOB_INDEX}.json"
NASDAQLIST = "nasdaqlisted.txt"

SYMBOL_RE = re.compile(r'^[A-Z0-9\.\-]{1,10}$')

# --- downloader & ticker list helpers (kept from previous robust version) ---
def ensure_nasdaq_list():
    if Path(NASDAQLIST).exists():
        print(f"[worker] Found {NASDAQLIST} locally.")
        return True

    print("[worker] nasdaqlisted.txt missing — trying to run download_nasdaq_list.py")
    if Path("download_nasdaq_list.py").exists():
        try:
            res = subprocess.run([sys.executable, "download_nasdaq_list.py"], capture_output=True, text=True, timeout=120)
            print("[worker] download_nasdaq_list.py stdout (truncated):")
            print(res.stdout[:2000])
            if res.stderr:
                print("[worker] download_nasdaq_list.py stderr (truncated):")
                print(res.stderr[:2000])
        except Exception as e:
            print("[worker] Running download script failed:", e)
            traceback.print_exc()

    if Path(NASDAQLIST).exists():
        print("[worker] nasdaqlisted.txt now present after running download script.")
        return True

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

def load_tickers(filename=NASDAQLIST):
    tickers = []
    try:
        with open(filename, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                low=line.lower()
                if low.startswith("symbol") or "file creation" in low or "nasdaqlisted" in low or line.startswith("#"):
                    continue
                parts = line.split("|")
                if not parts:
                    continue
                sym = parts[0].strip().upper()
                if SYMBOL_RE.match(sym):
                    tickers.append(sym)
    except Exception as e:
        print("[worker] load_tickers error:", e)
        traceback.print_exc()
    tickers = sorted(list(dict.fromkeys(tickers)))
    return tickers

# --- chunking ---
def chunk_list_round_robin(lst, total, index):
    if total <= 1:
        return lst[:]
    return [t for i,t in enumerate(lst) if (i % total) == index]

def chunk_list_block(lst, total, index):
    if total <= 1:
        return lst[:]
    n = len(lst)
    chunk_size = math.ceil(n / total)
    start = index * chunk_size
    end = min(start + chunk_size, n)
    return lst[start:end]

# --- safe conversions ---
def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

# --- STooq CSV fetch (fallback) ---
def fetch_stooq_csv(ticker, days=440):
    """
    ticker: plain symbol like AAPL or MSFT
    returns: pandas DataFrame with Date index and Open/High/Low/Close/Volume columns or None
    Stooq requires symbol appended with .US for US stocks.
    """
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    # Stooq expects uppercase and .US for U.S. stocks
    s = ticker.upper()
    if not s.endswith(".US"):
        s = s + ".US"
    url = f"https://stooq.com/q/d/l/?s={s}&d1={start.strftime('%Y%m%d')}&d2={end.strftime('%Y%m%d')}&i=d"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200 or not r.text:
            print(f"[stooq] {ticker} HTTP {r.status_code} empty response")
            return None
        df = pd.read_csv(StringIO(r.text), parse_dates=["Date"])
        if df.empty:
            return None
        df = df.rename(columns=lambda c: c.capitalize())
        df = df.set_index("Date").sort_index()
        # Ensure columns Open/High/Low/Close/Volume exist
        expected = {'Open','High','Low','Close','Volume'}
        if not expected.issubset(set(df.columns)):
            print(f"[stooq] {ticker} csv missing cols, got {list(df.columns)}")
            return None
        return df[['Open','High','Low','Close','Volume']]
    except Exception as e:
        print(f"[stooq] exception for {ticker}: {e}")
        return None

# --- yfinance fetch wrapper with retries & stooq fallback ---
def fetch_with_retries(ticker, retries=3, pause_base=1.0):
    last_exc = None
    # Try yfinance first
    for attempt in range(1, retries+1):
        try:
            df = yf.download(ticker, period="14mo", interval="1d", progress=False, threads=False)
            # sometimes yfinance returns an empty df; handle as failure
            if df is None or df.empty:
                raise ValueError("empty df")
            # Normalize columns
            df.columns = [c.capitalize() for c in df.columns]
            # Ensure expected columns
            if not {'Open','High','Low','Close','Volume'}.issubset(set(df.columns)):
                raise ValueError("missing columns from yfinance")
            # success
            return df[['Open','High','Low','Close','Volume']], "yfinance"
        except Exception as e:
            last_exc = e
            print(f"[fetch:yf] {ticker} attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(pause_base * (2 ** (attempt-1)))
    # yfinance failed -> try Stooq fallback once
    print(f"[fetch] yfinance failed for {ticker}; trying Stooq fallback")
    st_df = fetch_stooq_csv(ticker, days=440)
    if st_df is not None and not st_df.empty:
        return st_df, "stooq"
    # final failure
    raise last_exc if last_exc is not None else RuntimeError("unknown fetch failure")

# --- indicator computations (same as prior robust version) ---
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

        # RSI
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

        vol20 = vol.rolling(window=20).mean()
        out['avg_vol20'] = safe_float(vol20.iloc[-1], safe_float(vol.iloc[-1], 0.0))
        out['vol_spike'] = int(safe_float(vol.iloc[-1], 0.0) > VOL_SPIKE_MULT * out['avg_vol20'])

        # wave_strength heuristic
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
        traceback.print_exc()
        return None

# --- main ---
def main():
    errors = []
    results = []
    attempted = 0
    processed = 0
    yfinance_success = 0
    stooq_success = 0

    print(f"[worker] Starting job index={JOB_INDEX} total={JOB_TOTAL}")

    # ensure nasdaq list
    ok = ensure_nasdaq_list()
    if not ok:
        errors.append("nasdaqlisted.txt missing and download fallback failed.")
        print("[worker] ERROR: nasdaqlisted.txt missing. Wrote empty artifact and exiting.")
        out = {"results": [], "attempted_count": 0, "processed_count": 0, "errors": errors, "job_index": JOB_INDEX, "ts": datetime.utcnow().isoformat()+"Z"}
        Path(".").mkdir(parents=True, exist_ok=True)
        with open(OUT_FN, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        sys.exit(0)

    # read sample lines for debugging
    try:
        with open(NASDAQLIST, "r", encoding="utf-8", errors="ignore") as fh:
            sample_lines = [next(fh).rstrip("\n") for _ in range(10)]
        print("[worker] sample nasdaqlisted.txt first 10 lines:")
        for l in sample_lines:
            print("  ", l)
    except Exception:
        print("[worker] could not read sample lines of nasdaqlisted.txt (maybe short file)")

    tickers = load_tickers()
    print(f"[worker] Parsed {len(tickers)} tickers from {NASDAQLIST}")

    if not tickers:
        errors.append("Parsed 0 tickers from nasdaqlisted.txt")
        out = {"results": [], "attempted_count": 0, "processed_count": 0, "errors": errors, "job_index": JOB_INDEX, "ts": datetime.utcnow().isoformat()+"Z"}
        Path(".").mkdir(parents=True, exist_ok=True)
        with open(OUT_FN, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print("[worker] Wrote empty artifact because no tickers parsed. Check nasdaqlisted.txt contents.")
        sys.exit(0)

    # health check: try AAPL via yfinance to see if yfinance works
    try:
        print("[worker] performing AAPL health check via yfinance")
        try_df = yf.download("AAPL", period="1mo", interval="1d", progress=False, threads=False)
        print(f"[worker] AAPL health check - yfinance df type: {type(try_df)}, shape: {getattr(try_df,'shape',None)}")
        if try_df is None or getattr(try_df,'shape', (0,))[0] == 0:
            print("[worker] AAPL returned empty from yfinance -> yfinance may be blocked or rate-limited in this runner")
        else:
            print("[worker] AAPL returned data via yfinance (OK)")
    except Exception as e:
        print("[worker] AAPL health check exception:", e)

    # assign chunk (round-robin preferred)
    assigned = chunk_list_round_robin(tickers, JOB_TOTAL, JOB_INDEX)
    if not assigned:
        assigned = chunk_list_block(tickers, JOB_TOTAL, JOB_INDEX)
    print(f"[worker] Assigned {len(assigned)} tickers to this job. Sample: {assigned[:10]}")

    attempted = len(assigned)
    for i, ticker in enumerate(assigned, 1):
        try:
            print(f"[worker] ({i}/{attempted}) Fetching {ticker}")
            df, source = fetch_with_retries(ticker, retries=3)
            if df is None or df.empty or df.shape[0] < 30:
                # treat as skip but continue
                print(f"[worker] ({i}/{attempted}) {ticker} - insufficient data from {source}, skipping")
                errors.append(f"{ticker}: insufficient data from {source}")
                continue
            # count source usage
            if source == "yfinance":
                yfinance_success += 1
            elif source == "stooq":
                stooq_success += 1

            # drop NA and keep needed columns
            df = df[['Open','High','Low','Close','Volume']].dropna()
            indicators = compute_indicators(df)
            if indicators is None:
                print(f"[worker] ({i}/{attempted}) {ticker} - indicators None, skipping")
                errors.append(f"{ticker}: indicators None")
                continue
            indicators['ticker'] = ticker
            indicators['ts'] = datetime.utcnow().isoformat() + "Z"
            indicators['fetch_source'] = source
            results.append(indicators)
            processed += 1
            # polite short sleep
            time.sleep(0.06)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[worker] Exception processing {ticker}: {e}\n{tb}")
            errors.append(f"{ticker}: {str(e)}")
            continue

    # summary logs
    print(f"[worker] Fetch summary: yfinance_success={yfinance_success} stooq_success={stooq_success} processed={processed} attempted={attempted} errors={len(errors)}")

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
