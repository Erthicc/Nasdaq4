# worker.py
"""
Resilient worker: processes a chunk of NASDAQ tickers and ALWAYS writes
raw-results-{JOB_INDEX}.json (even on error). Designed so missing/failed
workers don't cancel the entire matrix.

Outputs a JSON with keys:
 - results: list of per-ticker indicator dicts
 - attempted_count: how many tickers this worker attempted
 - processed_count: how many tickers produced indicators
 - errors: list of error messages / tracebacks (strings)
 - job_index: JOB_INDEX
"""
import os
import sys
import json
import math
import traceback
import subprocess
from datetime import datetime
from pathlib import Path

# Defensive imports
try:
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import pandas_ta as ta
    import requests
except Exception as e:
    print("[worker] IMPORT ERROR:", e)
    traceback.print_exc()
    # If imports fail, still write an artifact saying nothing processed
    JOB_INDEX = int(os.environ.get("JOB_INDEX", "0"))
    out = {
        "results": [],
        "attempted_count": 0,
        "processed_count": 0,
        "errors": [f"IMPORT ERROR: {str(e)}"],
        "job_index": JOB_INDEX,
        "ts": datetime.utcnow().isoformat() + "Z"
    }
    Path(".").mkdir(parents=True, exist_ok=True)
    with open(f"raw-results-{JOB_INDEX}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    # Exit 0 so other workers continue; aggregator will detect empty results.
    sys.exit(0)

# Config
VOL_SPIKE_MULT = float(os.environ.get("VOL_SPIKE_MULT", "1.5"))
RECENT_DAYS_CROSSOVER = int(os.environ.get("RECENT_DAYS_CROSSOVER", "8"))
HISTORY_DAYS_FOR_SLOPE = int(os.environ.get("HISTORY_DAYS_FOR_SLOPE", "14"))
HISTORY_CHECK = HISTORY_DAYS_FOR_SLOPE + 1

JOB_TOTAL = int(os.environ.get("JOB_TOTAL", "1"))
JOB_INDEX = int(os.environ.get("JOB_INDEX", "0"))
OUT_FN = f"raw-results-{JOB_INDEX}.json"
NASDAQLIST = "nasdaqlisted.txt"

def ensure_nasdaq_list():
    if Path(NASDAQLIST).exists():
        return True
    # Try to run download script if present
    if Path("download_nasdaq_list.py").exists():
        try:
            subprocess.run([sys.executable, "download_nasdaq_list.py"], check=False, timeout=120)
        except Exception:
            pass
    # HTTP fallback
    try:
        url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and r.content:
            Path(NASDAQLIST).write_bytes(r.content)
            return True
    except Exception:
        pass
    return Path(NASDAQLIST).exists()

def load_tickers(filename=NASDAQLIST):
    tickers = []
    try:
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
    except Exception:
        return []
    for line in lines:
        if not line or line.strip().startswith("NASDAQ") or line.lower().startswith("symbol|"):
            continue
        parts = line.split('|')
        if not parts:
            continue
        sym = parts[0].strip()
        if not sym or any(tok in sym.lower() for tok in ("file", "creation", "nasdaqlisted")):
            continue
        tickers.append(sym)
    return sorted(list(dict.fromkeys(tickers)))

def chunk_list(lst, total, index):
    if total <= 1:
        return lst[:]
    n = len(lst)
    chunk_size = math.ceil(n / total)
    s = index * chunk_size
    return lst[s:s+chunk_size]

def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def compute_indicators(df):
    try:
        if df is None or df.empty or 'Close' not in df.columns:
            return None
        df = df.copy()
        df.columns = [c.capitalize() for c in df.columns]
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
        out['macd_slope'] = safe_float((macd_hist.iloc[-1] - macd_hist.iloc[-(HISTORY_DAYS_FOR_SLOPE+1)])/HISTORY_DAYS_FOR_SLOPE, 0.0) if len(macd_hist)>=HISTORY_DAYS_FOR_SLOPE+1 else 0.0
        m_recent = macd_line.tail(RECENT_DAYS_CROSSOVER); s_recent = signal.tail(RECENT_DAYS_CROSSOVER)
        out['macd_bull'] = int((m_recent> s_recent).any() and (m_recent.iloc[-1] > s_recent.iloc[-1]) if len(m_recent)>0 and len(s_recent)>0 else False)

        # RSI
        delta = close.diff()
        up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
        roll_up = up.rolling(window=14).mean(); roll_down = down.rolling(window=14).mean()
        rs = roll_up/(roll_down+1e-9); rsi = 100 - (100/(1+rs))
        out['rsi'] = safe_float(rsi.iloc[-1], 50.0)
        out['rsi_slope'] = safe_float((rsi.iloc[-1] - rsi.iloc[-(HISTORY_DAYS_FOR_SLOPE+1)])/HISTORY_DAYS_FOR_SLOPE, 0.0) if len(rsi)>=HISTORY_DAYS_FOR_SLOPE+1 else 0.0

        # SMA/EMA
        sma20 = close.rolling(window=20).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        out['sma20'] = safe_float(sma20.iloc[-1], safe_float(close.iloc[-1],0.0))
        out['ema50'] = safe_float(ema50.iloc[-1], safe_float(close.iloc[-1],0.0))
        out['above_trend'] = int((close.iloc[-1] > out['sma20']) and (close.iloc[-1] > out['ema50']))

        # Bollinger
        std20 = close.rolling(window=20).std()
        out['bb_breakout'] = int(close.iloc[-1] > safe_float((sma20+2*std20).iloc[-1], 1e12))

        # ADX
        try:
            adx_df = ta.adx(high=high, low=low, close=close, length=14)
            adx_col = next((c for c in adx_df.columns if str(c).upper().startswith("ADX")), None)
            out['adx'] = safe_float(adx_df[adx_col].iloc[-1], 0.0) if adx_col else 0.0
        except Exception:
            out['adx'] = 0.0

        # ATR
        try:
            atr = ta.atr(high=high, low=low, close=close, length=14)
            out['atr'] = safe_float(atr.iloc[-1], 0.0)
        except Exception:
            out['atr'] = safe_float((high-low).rolling(14).mean().iloc[-1], 0.0)

        # OBV slope
        try:
            obv = ta.obv(close=close, volume=vol)
            out['obv_slope'] = safe_float((obv.iloc[-1] - obv.iloc[-(HISTORY_DAYS_FOR_SLOPE+1)])/HISTORY_DAYS_FOR_SLOPE, 0.0) if len(obv)>=HISTORY_DAYS_FOR_SLOPE+1 else 0.0
        except Exception:
            out['obv_slope'] = 0.0

        vol20 = vol.rolling(window=20).mean()
        out['avg_vol20'] = safe_float(vol20.iloc[-1], safe_float(vol.iloc[-1],0.0))
        out['vol_spike'] = int(safe_float(vol.iloc[-1],0.0) > VOL_SPIKE_MULT * out['avg_vol20'])

        # wave_strength heuristic
        try:
            recent_close = close.tail(60)
            peaks = recent_close[(recent_close.shift(1) < recent_close) & (recent_close.shift(-1) < recent_close)]
            out['wave_strength'] = safe_float(peaks.iloc[-1]/out['sma20'], 1.0) if len(peaks)>0 and out['sma20']>0 else 1.0
        except Exception:
            out['wave_strength'] = 1.0

        out['last_close'] = safe_float(close.iloc[-1],0.0)
        return out
    except Exception:
        traceback_str = traceback.format_exc()
        print("[compute_indicators] exception:", traceback_str)
        return None

def main():
    errors = []
    results = []
    attempted = 0
    processed = 0

    try:
        ok = ensure_nasdaq_list()
        if not ok:
            errors.append("nasdaqlisted.txt not present and download failed.")
    except Exception:
        errors.append("Exception while ensuring nasdaq list: " + traceback.format_exc())

    tickers = []
    try:
        tickers = load_tickers()
    except Exception:
        errors.append("Exception while loading tickers: " + traceback.format_exc())

    assigned = []
    try:
        assigned = chunk_list(tickers, JOB_TOTAL, JOB_INDEX)
    except Exception:
        errors.append("Exception while chunking tickers: " + traceback.format_exc())

    attempted = len(assigned)
    for i, ticker in enumerate(assigned, 1):
        try:
            # fetch data
            df = yf.download(ticker, period="14mo", interval="1d", progress=False, threads=False)
            if df is None or df.empty or df.shape[0] < 30:
                # skip but count as attempted
                continue
            df = df[['Open','High','Low','Close','Volume']].dropna()
            indicators = compute_indicators(df)
            if indicators is None:
                continue
            indicators['ticker'] = ticker
            indicators['ts'] = datetime.utcnow().isoformat() + "Z"
            results.append(indicators)
            processed += 1
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[worker] Exception for {ticker}: {e}\n{tb}")
            errors.append(f"{ticker}: {str(e)}\n{tb}")
            continue

    # Always write an artifact, even if empty or errors occurred
    out = {
        "results": results,
        "attempted_count": attempted,
        "processed_count": processed,
        "errors": errors,
        "job_index": JOB_INDEX,
        "ts": datetime.utcnow().isoformat() + "Z"
    }

    try:
        Path(".").mkdir(parents=True, exist_ok=True)
        with open(OUT_FN, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print(f"[worker] WROTE {OUT_FN} - attempted={attempted} processed={processed} errors={len(errors)}")
    except Exception as e:
        print("[worker] Failed to write artifact:", e)
        print(traceback.format_exc())
        # try to write minimal fallback file
        try:
            with open(OUT_FN, "w", encoding="utf-8") as fh:
                json.dump({"results": [], "attempted_count": attempted, "processed_count": processed, "errors": ["write failed: "+str(e)]}, fh)
        except Exception:
            pass

    # Exit 0 so other workers and aggregator continue.
    sys.exit(0)

if __name__ == "__main__":
    main()
