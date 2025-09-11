#!/usr/bin/env python3
# finalize.py
"""
NASDAQ Analyzer finalizer — revised scoring to prefer stocks that are likely
to ramp sustainably instead of immediate pop-and-crash tickers.

Key scoring changes:
- RSI: penalize RSI > 50; reward RSI <= 40 (oversold). RSI in 40-50 is neutral.
- Volume: prefer medium-low liquidity (bell-shaped preference peaking ~25th percentile).
- Volume spikes (vol_spike): penalized (negative contribution).
- Momentum (mom14): prefer moderate positive momentum; penalize extreme momentum.
- ATR: inverted (lower volatility preferred).
- Other signals (MACD bullish crossover, MACD slope, OBV slope) still help but
  we avoid over-weighting explosive single-day signals.

Drop-in: Replace your existing finalize.py with this file.
"""

import glob
import json
import os
import math
import time
import traceback
from datetime import datetime
from pathlib import Path

# optional imports for history fetching
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

# numeric features expected in worker outputs (raw)
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
    "macd_bull", "bb_breakout", "vol_spike", "above_trend", "avg_vol20"
]

# Weights: tune these as needed.
# Note: vol_spike is negative (penalty). avg_vol20 included with positive weight,
# but avg_vol20 will be transformed to prefer medium-low liquidity.
WEIGHTS = {
    "macd_hist": 2.0,
    "macd_slope": 2.0,
    "rsi": 3.0,            # transformed: positive for RSI<=40, negative for RSI>50
    "rsi_slope": 0.6,
    "wave_strength": 1.8,
    "adx": 0.8,
    "atr": 1.0,            # inverted so lower ATR is better
    "obv_slope": 1.5,
    "mom14": 1.2,          # transformed to prefer moderate momentum
    "macd_bull": 3.0,
    "bb_breakout": 0.6,
    "vol_spike": -1.8,     # penalize volume spikes strongly
    "above_trend": 0.8,
    "avg_vol20": 1.0       # uses bell-shaped transform (preferred mid-low)
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
        return [0.5 for _ in xs]
    return [(x - mn) / (mx - mn) for x in xs]

def ensure_dirs():
    Path(PUBLIC_DIR).mkdir(parents=True, exist_ok=True)
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(ARCHIVE_DIR).mkdir(parents=True, exist_ok=True)

def build_explanation(row):
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
            expl.append("recent volume spike (penalized)")
    except Exception:
        pass
    try:
        rsi_val = float(row.get("rsi", 50))
        if rsi_val <= 40:
            expl.append("RSI oversold (preferred)")
        elif rsi_val > 50:
            expl.append("RSI above 50 (penalized)")
    except Exception:
        pass
    try:
        if float(row.get("wave_strength", 1)) > 1.05:
            expl.append("strong wave vs SMA20")
    except Exception:
        pass
    try:
        if float(row.get("obv_slope", 0)) > 0:
            expl.append("rising OBV (confirmation)")
    except Exception:
        pass
    if not expl:
        return "no significant signals"
    return "; ".join(expl)

def transform_rsi_preference(rsi_raw):
    """
    Transform raw RSI so that:
    - RSI <= 40 -> positive [0..1] (40->0, 0->1)
    - RSI 40..50 -> neutral (0)
    - RSI > 50 -> negative penalty (mapped -0..-1, e.g., 50->0, 100->-1)
    """
    r = max(0.0, min(100.0, safef(rsi_raw)))
    if r <= 40.0:
        return (40.0 - r) / 40.0  # 0..1
    if r <= 50.0:
        return 0.0
    # penalty for >50
    return -((r - 50.0) / 50.0)  # -0..-1

def transform_mom_preference(norm_mom):
    """
    Prefer moderate momentum. norm_mom is 0..1 percentile of mom14 (after min-max).
    We shape a peaked function centered near 0.6 (moderate positive momentum).
    """
    x = max(0.0, min(1.0, norm_mom))
    # peaked quadratic shape centered at 0.6 with width ~0.4:
    center = 0.6
    width = 0.4
    val = 1.0 - ((x - center) / width) ** 2
    return max(0.0, val)  # clip to [0,1]

def transform_vol_pref(norm_vol):
    """
    Prefer medium-low volume. norm_vol is 0..1 percentile of avg_vol20.
    Peak near 0.25 (25th percentile) and fall off to 0 at extreme low/high.
    """
    x = max(0.0, min(1.0, norm_vol))
    preferred = 0.25
    span = 0.25
    score = 1.0 - (abs(x - preferred) / span)
    return max(0.0, min(1.0, score))

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

    # collect numeric and boolean arrays
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
            "rsi_raw": safef(r.get("rsi", 50.0)),
            "mom14_raw": safef(r.get("mom14", 0.0))
        })

    # Normalize numeric features to [0,1]
    normalized = {}
    for f, arr in numeric_vals.items():
        normalized[f] = min_max_list(arr)

    # Transform RSI into preference/penalty (use raw RSI from numeric_vals before normalization)
    # We'll compute rsi_pref_list corresponding to each row
    rsi_raws = [safef(x) for x in numeric_vals.get('rsi', [])]
    rsi_pref_list = [transform_rsi_preference(x) for x in rsi_raws]
    # We'll normalize rsi_pref_list into 0..1 for scoring where positive contributions map to 0..1 and negative to -1..0.
    # But to keep aggregation consistent, we will map rsi_pref_list from [-1..1] to [-1..1] (keep as-is).
    # For later combination we place this under normalized['rsi'] but allow negatives by shifting base.
    # We'll create a separate list stored in normalized_rsi_pref (values in [-1..1])
    normalized_rsi_pref = rsi_pref_list  # may contain negatives

    # ATR invert: lower ATR is better (we already normalized atr above)
    if 'atr' in normalized:
        normalized['atr'] = [1.0 - v for v in normalized['atr']]

    # Transform avg_vol20 to prefer medium-low liquidity
    if 'avg_vol20' in normalized:
        vol_norm = normalized['avg_vol20']
        vol_pref = [transform_vol_pref(v) for v in vol_norm]
        normalized['avg_vol20'] = vol_pref

    # Transform mom14 to prefer moderate momentum
    if 'mom14' in normalized:
        mom_norm = normalized['mom14']
        mom_pref = [transform_mom_preference(v) for v in mom_norm]
        normalized['mom14'] = mom_pref

    # If macd_hist negative values could be present; we already normalized macd_hist to 0..1 via min_max_list.
    # Keep other normalized features as-is.

    # Prepare items list with computed raw composite
    items = []
    # Adjust feature list mapping: use normalized values, but for RSI use transformed pref which can be negative.
    abs_sum = sum(abs(WEIGHTS.get(k, 0)) for k in ALL_FEATURES) or 1.0

    for i, r in enumerate(rows):
        # build feature vector in ALL_FEATURES order
        feat_vector = []
        for feat_name in ALL_FEATURES:
            if feat_name == 'rsi':
                # use transformed rsi pref mapped into -1..1
                val = normalized_rsi_pref[i] if i < len(normalized_rsi_pref) else 0.0
                # For aggregation, to keep scales similar to other 0..1 features, leave val in [-1..1]
                feat_vector.append(val)
            else:
                arr = normalized.get(feat_name, None)
                if arr is None:
                    # if a boolean feature or missing numeric, try bools
                    if feat_name in BOOL_FEATURES:
                        b_arr = bool_vals.get(feat_name, [0] * len(rows))
                        val = float(b_arr[i]) if i < len(b_arr) else 0.0
                        feat_vector.append(val)
                    else:
                        feat_vector.append(0.0)
                else:
                    val = arr[i] if i < len(arr) else 0.0
                    feat_vector.append(val)

        # compute raw composite (weights may be negative)
        raw = 0.0
        for val, feat_name in zip(feat_vector, ALL_FEATURES):
            w = WEIGHTS.get(feat_name, 0.0)
            raw += val * w

        composite = raw / abs_sum
        features_raw = {f: (numeric_vals.get(f, [0]*len(rows))[i] if f in numeric_vals else None) for f in NUMERIC_FEATURES}
        bools_raw = {b: (bool_vals.get(b, [0]*len(rows))[i] if b in bool_vals else 0) for b in BOOL_FEATURES}
        items.append({
            "ticker": r.get("ticker"),
            "raw": composite,
            "features_raw": features_raw,
            "bools_raw": bools_raw,
            "extras": extras[i]
        })

    # Normalize composite to 0..1 for stable mapping
    raw_vals = [it['raw'] for it in items]
    mn_raw = min(raw_vals); mx_raw = max(raw_vals)
    if math.isclose(mn_raw, mx_raw):
        for it in items:
            it['score01'] = 0.5
    else:
        for it in items:
            # linear map raw in [mn_raw,mx_raw] -> [0,1]
            it['score01'] = (it['raw'] - mn_raw) / (mx_raw - mn_raw)

    # produce final fields and human explanations
    for it in items:
        it['score_0_100'] = round(it['score01'] * 100, 2)
        it['score_0_10'] = round(it['score01'] * 10, 2)
        merged = {}
        merged.update(it.get('features_raw', {}))
        merged.update(it.get('bools_raw', {}))
        merged['rsi'] = it['extras'].get('rsi_raw', None)
        merged['avg_vol20'] = it['extras'].get('avg_vol20', None)
        merged['last_close'] = it['extras'].get('last_close', None)
        merged['mom14'] = it['extras'].get('mom14_raw', None)
        it['explanation'] = build_explanation(merged)

    # Sorting: primary score, secondary obv_slope, tertiary prefer moderate liquidity (avg_vol20)
    def tiebreaker_key(it):
        obv = it['features_raw'].get('obv_slope', 0) or 0
        # prefer avg_vol20 transformed (higher better because we converted to preference)
        avg_vol_pref_list = normalized.get('avg_vol20', [0]*len(items))
        avg_vol_pref = avg_vol_pref_list[0] if not avg_vol_pref_list else avg_vol_pref_list[0]
        # Primary: score_0_100, secondary obv slope, tertiary avg_vol pref, last_close as last tie-break
        return (it['score_0_100'], obv, it['features_raw'].get('avg_vol20', 0), it['extras'].get('last_close', 0))

    items.sort(key=lambda it: (it['score_0_100'],
                               it['features_raw'].get('obv_slope', 0),
                               it['features_raw'].get('avg_vol20', 0),
                               it['extras'].get('last_close', 0)),
               reverse=True)

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
            "avg_vol20": it.get('features_raw', {}).get('avg_vol20'),
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

    # Build items_map for per-ticker JSON files
    items_map = {}
    for it in items:
        items_map[it['ticker']] = {
            "score_0_100": it['score_0_100'],
            "score_0_10": it['score_0_10'],
            "explanation": it['explanation'],
            **it.get('features_raw', {}),
            **it.get('bools_raw', {})
        }

    # Generate per-ticker JSON files for top N (and refresh stale ones)
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
        success_in_batch = set()
        for attempt in range(1, FETCH_RETRIES + 1):
            try:
                print(f"[finalize] batch download attempt {attempt} for {len(batch)} tickers...")
                df = yf.download(batch, period=f"{HISTORY_DAYS}d", interval="1d", group_by='ticker', threads=False, progress=False)
                if df is None or df.empty:
                    raise RuntimeError("yfinance returned empty dataframe for batch")
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
