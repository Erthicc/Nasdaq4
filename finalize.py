#!/usr/bin/env python3
# finalize.py
"""
NASDAQ Analyzer finalizer (copy-paste into repo root).

Reads worker artifact raw-results-*.json files, aggregates indicator rows,
applies min-max normalization, transforms RSI to favor undervalued (RSI <= 40),
computes weighted composite scores, writes public/top_picks.json,
archives snapshots, and writes per-ticker JSON files (indicators + optional OHLCV history).

Dependencies (requirements.txt): yfinance, pandas, numpy, ta (if used in worker)
"""

import glob
import json
import os
import math
import time
import traceback
from datetime import datetime
from pathlib import Path

# try imports for fetching histories
try:
    import yfinance as yf
    import pandas as pd
except Exception as e:
    print("Missing module (yfinance/pandas). Please install requirements. Error:", e)
    raise

# -------------------- Config --------------------
PUBLIC_DIR = "public"
DATA_DIR = os.path.join(PUBLIC_DIR, "data")
ARCHIVE_DIR = os.path.join(DATA_DIR, "archive")
TOP_JSON = os.path.join(PUBLIC_DIR, "top_picks.json")

# numeric features expected in worker outputs
NUMERIC_FEATURES = [
    "macd_hist", "macd_slope", "rsi", "rsi_slope",
    "wave_strength", "adx", "atr", "obv_slope", "mom14", "avg_vol20", "last_close"
]

# boolean features (0/1)
BOOL_FEATURES = [
    "macd_bull", "bb_breakout", "vol_spike", "above_trend"
]

# All features used for weighted aggregation (order matters)
ALL_FEATURES = [
    "macd_hist", "macd_slope", "rsi", "rsi_slope",
    "wave_strength", "adx", "atr", "obv_slope", "mom14",
    "macd_bull", "bb_breakout", "vol_spike", "above_trend"
]

# Weights: tune these as you like. Positive weights mean higher normalized feature -> better.
WEIGHTS = {
    "macd_hist": 2.0,
    "macd_slope": 1.5,
    "rsi": 2.0,            # NOTE: RSI is transformed to reward RSI<=40
    "rsi_slope": 0.8,
    "wave_strength": 2.0,
    "adx": 1.0,
    "atr": 1.0,            # ATR will be inverted (lower ATR -> better)
    "obv_slope": 1.0,
    "mom14": 1.2,
    "macd_bull": 3.0,
    "bb_breakout": 0.9,
    "vol_spike": 0.8,
    "above_trend": 1.0
}

# History fetch configuration (per-ticker OHLCV for chart pages)
HISTORY_DAYS = 440       # ~14 months
HISTORY_FOR_TOP_N = 250  # fetch histories only for top N tickers to limit requests
YF_BATCH_SIZE = 50       # number of tickers per batch in yfinance.download
YF_BATCH_PAUSE = 2.0     # seconds pause between batches
REFRESH_DAYS = 7         # refresh per-ticker JSON only if older than this
FETCH_RETRIES = 3

# ------------------------------------------------

def safef(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def find_artifacts():
    files = sorted(glob.glob("**/raw-results-*.json", recursive=True))
    return files

def load_json(fn):
    try:
        with open(fn, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        print("Failed to load", fn, e)
        return None

def min_max_list(xs):
    if not xs:
        return []
    mn = min(xs); mx = max(xs)
    if math.isclose(mn, mx):
        # avoid divide-by-zero; give neutral mid value
        return [0.5 for _ in xs]
    return [(x - mn) / (mx - mn) for x in xs]

def ensure_dirs():
    Path(PUBLIC_DIR).mkdir(parents=True, exist_ok=True)
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(ARCHIVE_DIR).mkdir(parents=True, exist_ok=True)

def build_explanation(row):
    # simple english explanation builder from indicators; row contains raw (not normalized) features
    expl = []
    try:
        if int(row.get("macd_bull", 0)):
            expl.append("recent MACD bullish crossover")
    except Exception:
        pass
    try:
        if int(row.get("bb_breakout", 0)):
            expl.append("Bollinger upper-band breakout")
    except Exception:
        pass
    try:
        if float(row.get("adx", 0)) > 25:
            expl.append("strong ADX trend")
    except Exception:
        pass
    try:
        if int(row.get("vol_spike", 0)):
            expl.append("volume spike")
    except Exception:
        pass
    try:
        if float(row.get("rsi", 50)) < 30:
            expl.append("RSI oversold")
        elif float(row.get("rsi", 50)) > 70:
            expl.append("RSI overbought")
    except Exception:
        pass
    try:
        if float(row.get("wave_strength", 1)) > 1.05:
            expl.append("strong wave vs SMA20")
    except Exception:
        pass
    try:
        if float(row.get("obv_slope", 0)) > 0:
            expl.append("rising OBV")
    except Exception:
        pass
    if not expl:
        return "no significant signals"
    return "; ".join(expl)

def aggregate():
    ensure_dirs()
    files = find_artifacts()
    print(f"[finalize] found {len(files)} artifact files")
    total_attempted = 0
    total_processed = 0
    errors = []
    rows = []

    for f in files:
        j = load_json(f)
        if not j:
            continue
        total_attempted += int(j.get("attempted_count", 0))
        total_processed += int(j.get("processed_count", 0))
        for e in j.get("errors", []):
            errors.append(f"{f}: {e}")
        for r in j.get("results", []):
            # normalize keys, ensure ticker present
            if not r.get("ticker"):
                continue
            rows.append(r)

    print(f"[finalize] attempted={total_attempted}, processed={total_processed}, rows={len(rows)}, errors={len(errors)}")
    if not rows:
        out = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "count_total": total_attempted,
            "count_results": total_processed,
            "failed_count": max(0, total_attempted - total_processed),
            "errors": errors,
            "top": []
        }
        with open(TOP_JSON, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print("[finalize] wrote empty top_picks.json")
        return out

    # prepare arrays for normalization
    numeric_vals = {f: [] for f in NUMERIC_FEATURES}
    bool_vals = {b: [] for b in BOOL_FEATURES}
    extras = []

    for r in rows:
        for f in NUMERIC_FEATURES:
            numeric_vals[f].append(safef(r.get(f, 0.0)))
        for b in BOOL_FEATURES:
            bool_vals[b].append(1 if int(r.get(b, 0)) else 0)
        extras.append({
            "ticker": r.get("ticker"),
            "avg_vol20": safef(r.get("avg_vol20", 0.0)),
            "last_close": safef(r.get("last_close", 0.0)),
            "rsi_raw": safef(r.get("rsi", 50.0))
        })

    # Transform RSI to prefer undervalued (RSI <= 40) — your requested change
    if 'rsi' in numeric_vals:
        transformed = []
        for r in numeric_vals['rsi']:
            # clamp between 0..100
            rclamp = max(0.0, min(100.0, r))
            if rclamp <= 40.0:
                # map 0->1.0, 40->0.0 linearly
                transformed.append((40.0 - rclamp) / 40.0)
            else:
                transformed.append(0.0)
        numeric_vals['rsi'] = transformed
        print("[finalize] transformed RSI (undersold preference) sample:", numeric_vals['rsi'][:8])

    # Normalize numeric features to [0,1]
    normalized = {}
    for f, arr in numeric_vals.items():
        normalized[f] = min_max_list(arr)

    # invert ATR so lower volatility is better
    if 'atr' in normalized:
        normalized['atr'] = [1.0 - v for v in normalized['atr']]

    # Prepare items list with computed raw composite
    items = []
    abs_sum = sum(abs(WEIGHTS.get(k, 0)) for k in ALL_FEATURES) or 1.0

    # build list of item dicts
    for i, r in enumerate(rows):
        feat_values = []
        # numeric order must match ALL_FEATURES order for numeric features
        for f in NUMERIC_FEATURES:
            # handle missing normalized arrays (fallback to 0)
            arr = normalized.get(f, [0.0] * len(rows))
            val = arr[i] if i < len(arr) else 0.0
            feat_values.append(val)
        # bools appended in same order as BOOL_FEATURES, but ALL_FEATURES expects bools after numeric in earlier definition
        bool_vector = []
        for b in BOOL_FEATURES:
            b_arr = bool_vals.get(b, [0] * len(rows))
            bval = b_arr[i] if i < len(b_arr) else 0
            bool_vector.append(float(bval))
        # combine numeric + bool in the same sequence that ALL_FEATURES lists
        # find index mapping: numeric features matching prefix of ALL_FEATURES, then bools
        combined_vector = []
        for feat_name in ALL_FEATURES:
            if feat_name in NUMERIC_FEATURES:
                # index in numeric features
                idx = NUMERIC_FEATURES.index(feat_name)
                combined_vector.append(feat_values[idx])
            else:
                # boolean feature
                idxb = BOOL_FEATURES.index(feat_name)
                combined_vector.append(bool_vector[idxb])

        # compute raw score
        raw = 0.0
        for val, feat_name in zip(combined_vector, ALL_FEATURES):
            raw += val * WEIGHTS.get(feat_name, 0.0)

        composite = raw / abs_sum
        items.append({
            "ticker": r.get("ticker"),
            "raw": composite,
            "combined_vector": combined_vector,
            "features_raw": {f: numeric_vals.get(f, [0]*len(rows))[i] if f in numeric_vals else None for f in NUMERIC_FEATURES},
            "bools_raw": {b: bool_vals.get(b, [0]*len(rows))[i] if b in bool_vals else 0 for b in BOOL_FEATURES},
            "extras": extras[i]
        })

    # Normalize composite to 0..1
    raw_vals = [it['raw'] for it in items]
    mn_raw = min(raw_vals); mx_raw = max(raw_vals)
    if math.isclose(mn_raw, mx_raw):
        for it in items:
            it['score01'] = 0.5
    else:
        for it in items:
            it['score01'] = (it['raw'] - mn_raw) / (mx_raw - mn_raw)

    # map to 0-100 and 0-10, build explanation
    for it in items:
        it['score_0_100'] = round(it['score01'] * 100, 2)
        it['score_0_10'] = round(it['score01'] * 10, 2)
        # Merge features for explanation builder (use raw numeric values where useful)
        merged = {}
        merged.update(it.get('features_raw', {}))
        merged.update(it.get('bools_raw', {}))
        # ensure RSI in explanation is the original raw RSI (not transformed)
        merged['rsi'] = it['extras'].get('rsi_raw', None)
        merged['avg_vol20'] = it['extras'].get('avg_vol20', None)
        merged['last_close'] = it['extras'].get('last_close', None)
        it['explanation'] = build_explanation(merged)

    # Sorting: primary composite score, secondary macd_hist (raw numeric before normalization), tertiary avg_vol20
    def tiebreaker_key(it):
        macd_hist_raw = it['features_raw'].get('macd_hist', 0) if it.get('features_raw') else 0
        avg_vol = it['extras'].get('avg_vol20', 0) if it.get('extras') else 0
        return (it['score_0_100'], macd_hist_raw, avg_vol)

    items.sort(key=tiebreaker_key, reverse=True)

    # Prepare top array to write
    top = []
    for it in items:
        top.append({
            "ticker": it['ticker'],
            "score_0_100": it['score_0_100'],
            "score_0_10": it['score_0_10'],
            "features": it.get('features_raw', {}),
            "bools": it.get('bools_raw', {}),
            "explanation": it.get('explanation', ''),
            "avg_vol20": it.get('extras', {}).get('avg_vol20'),
            "last_close": it.get('extras', {}).get('last_close')
        })

    out = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "count_total": total_attempted,
        "count_results": total_processed,
        "failed_count": max(0, total_attempted - total_processed),
        "errors": errors,
        "top": top
    }

    # write top_picks.json
    with open(TOP_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("[finalize] wrote", TOP_JSON)

    # archive snapshot
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    archive_fn = os.path.join(ARCHIVE_DIR, f"{ts}_top_picks.json")
    with open(archive_fn, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("[finalize] archived to", archive_fn)

    # Build items_map to include indicators for per-ticker JSON files
    items_map = {}
    for it in items:
        items_map[it['ticker']] = {
            "score_0_100": it['score_0_100'],
            "score_0_10": it['score_0_10'],
            "explanation": it['explanation'],
            **it.get('features_raw', {}),
            **it.get('bools_raw', {})
        }

    # Generate per-ticker JSON files: but only for top N (and refresh stale ones)
    top_tickers = [it['ticker'] for it in items[:HISTORY_FOR_TOP_N]]

    def file_age_days(path):
        try:
            m = Path(path).stat().st_mtime
            return (time.time() - m) / 86400.0
        except Exception:
            return 9999.0

    def save_ticker_json(ticker, indicators, history_records):
        obj = {
            "ticker": ticker,
            "indicators": indicators,
            "history": history_records or []
        }
        outp = os.path.join(DATA_DIR, f"{ticker}.json")
        try:
            with open(outp, "w", encoding="utf-8") as fh:
                json.dump(obj, fh, indent=2)
        except Exception as e:
            print(f"[finalize] Failed to write {outp}: {e}")

    # determine which tickers need fetching (missing or stale)
    to_fetch = []
    for t in top_tickers:
        p = os.path.join(DATA_DIR, f"{t}.json")
        if not os.path.exists(p):
            to_fetch.append(t)
        else:
            if file_age_days(p) >= REFRESH_DAYS:
                to_fetch.append(t)

    print(f"[finalize] need to fetch history for {len(to_fetch)} tickers (top {HISTORY_FOR_TOP_N})")

    # helper to chunk lists
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    remaining = set(to_fetch)
    # batch fetch via yfinance.download
    for batch in chunks(list(to_fetch), YF_BATCH_SIZE):
        tickers_str = " ".join(batch)
        success_in_batch = set()
        for attempt in range(1, FETCH_RETRIES + 1):
            try:
                print(f"[finalize] batch download attempt {attempt} for {len(batch)} tickers...")
                df = yf.download(batch, period=f"{HISTORY_DAYS}d", interval="1d", group_by='ticker', threads=False, progress=False)
                if df is None or df.empty:
                    raise RuntimeError("yfinance returned empty dataframe for batch")
                # If multi-index columns
                if isinstance(df.columns, pd.MultiIndex):
                    for t in batch:
                        try:
                            sub = df[t].dropna(how='all')
                            if sub is None or sub.empty:
                                continue
                            sub = sub.reset_index()
                            sub['Date'] = pd.to_datetime(sub['Date']).dt.strftime("%Y-%m-%d")
                            hist = sub[['Date','Open','High','Low','Close','Volume']].to_dict(orient='records')
                            indicators = items_map.get(t, {"score_0_100": None, "explanation": None})
                            save_ticker_json(t, indicators, hist)
                            success_in_batch.add(t)
                        except Exception as e:
                            print(f"[finalize] extraction failed for {t}: {e}")
                else:
                    # single ticker frame handling
                    for t in batch:
                        try:
                            sub = df.dropna(how='all')
                            if sub is None or sub.empty:
                                continue
                            sub = sub.reset_index()
                            sub['Date'] = pd.to_datetime(sub['Date']).dt.strftime("%Y-%m-%d")
                            hist = sub[['Date','Open','High','Low','Close','Volume']].to_dict(orient='records')
                            indicators = items_map.get(t, {"score_0_100": None, "explanation": None})
                            save_ticker_json(t, indicators, hist)
                            success_in_batch.add(t)
                        except Exception as e:
                            print(f"[finalize] single extraction failed for {t}: {e}")
                # break retry loop after success
                for s in success_in_batch:
                    if s in remaining:
                        remaining.discard(s)
                print(f"[finalize] batch succeeded for {len(success_in_batch)} tickers; remaining {len(remaining)}")
                break
            except Exception as e:
                print(f"[finalize] batch attempt {attempt} failed: {e}")
                time.sleep(YF_BATCH_PAUSE * attempt)
        time.sleep(YF_BATCH_PAUSE)

    # fallback per-ticker for remaining
    if remaining:
        print(f"[finalize] fallback per-ticker fetch for {len(remaining)} tickers")
    for t in list(remaining):
        for attempt in range(1, FETCH_RETRIES + 1):
            try:
                df = yf.Ticker(t).history(period=f"{HISTORY_DAYS}d", interval="1d", auto_adjust=False, actions=False)
                if df is None or df.empty:
                    raise RuntimeError("empty history")
                sub = df.reset_index()
                sub['Date'] = pd.to_datetime(sub['Date']).dt.strftime("%Y-%m-%d")
                hist = sub[['Date','Open','High','Low','Close','Volume']].to_dict(orient='records')
                indicators = items_map.get(t, {"score_0_100": None, "explanation": None})
                save_ticker_json(t, indicators, hist)
                remaining.discard(t)
                break
            except Exception as e:
                print(f"[finalize] per-ticker fetch {t} attempt {attempt} failed: {e}")
                time.sleep(YF_BATCH_PAUSE * attempt)
        if t in remaining:
            # write indicator-only JSON
            indicators = items_map.get(t, {})
            save_ticker_json(t, indicators, [])
            print(f"[finalize] wrote indicator-only JSON for {t} (no history)")

    print("[finalize] per-ticker generation complete. remaining:", len(remaining))
    return out

if __name__ == "__main__":
    try:
        aggregate()
        print("[finalize] done")
    except Exception as exc:
        print("[finalize] unhandled exception:", exc)
        traceback.print_exc()
        raise
