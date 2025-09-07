# finalize.py
import json
import glob
import os
from datetime import datetime
import numpy as np
import math

NUMERIC_FEATURES = ['macd_hist', 'macd_slope', 'rsi', 'rsi_slope',
                    'wave_strength', 'adx', 'atr', 'obv_slope']
BOOL_FEATURES = ['macd_bull', 'bb_breakout', 'vol_spike', 'above_trend']
ALL_FEATURES = NUMERIC_FEATURES + BOOL_FEATURES

WEIGHTS = {
    'macd_hist': 2.0, 'macd_slope': 2.0, 'rsi': 1.5, 'rsi_slope': 1.5,
    'wave_strength': 2.0, 'adx': 1.0, 'atr': 1.0, 'obv_slope': 1.0,
    'macd_bull': 3.0, 'bb_breakout': 1.0, 'vol_spike': 0.5, 'above_trend': 1.0
}

def load_worker_artifacts(pattern="raw-results-*.json"):
    files = sorted(glob.glob(pattern))
    artifacts = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                data['_source_file'] = f
                artifacts.append(data)
        except Exception as e:
            print(f"[finalize] Failed to load {f}: {e}")
    return artifacts

def min_max_scale(arr):
    arr = np.array(arr, dtype=float)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if math.isclose(mx, mn):
        return np.full_like(arr, 0.5)
    return (arr - mn) / (mx - mn)

def build_explanation(row):
    expl = []
    if int(row.get('macd_bull', 0)): expl.append("recent MACD bullish crossover")
    if int(row.get('bb_breakout', 0)): expl.append("Bollinger upper-band breakout")
    if row.get('adx', 0) and float(row.get('adx', 0)) > 25: expl.append(f"strong trend (ADX {row.get('adx'):.1f})")
    if int(row.get('vol_spike', 0)): expl.append("volume spike")
    if int(row.get('above_trend', 0)): expl.append("price above 50-day EMA & 20-day SMA")
    rsi = float(row.get('rsi', 50))
    if rsi > 70: expl.append("RSI overbought")
    elif rsi < 30: expl.append("RSI oversold")
    if float(row.get('obv_slope', 0)) > 0: expl.append("rising OBV")
    if float(row.get('wave_strength', 1)) > 1.05: expl.append("strong upward wave")
    return "; ".join(expl) if expl else "no significant signals"

def main():
    artifacts = load_worker_artifacts()
    if not artifacts:
        print("[finalize] No worker artifacts found - nothing to finalize.")
        return

    # aggregate counts & errors
    total_attempted = 0
    total_processed = 0
    all_errors = []
    rows = []  # per-ticker results (flattened)

    for art in artifacts:
        attempted = int(art.get("attempted_count", 0))
        processed = int(art.get("processed_count", 0))
        total_attempted += attempted
        total_processed += processed
        errs = art.get("errors", [])
        if errs:
            all_errors.extend([f"{art.get('_source_file','?')}: {e}" for e in errs])
        for r in art.get("results", []):
            rows.append(r)

    print(f"[finalize] worker artifacts loaded: files={len(artifacts)} attempted_total={total_attempted} processed_total={total_processed} errors={len(all_errors)}")

    if not rows:
        print("[finalize] No ticker rows to score; writing empty public/top_picks.json with metadata.")
        os.makedirs("public", exist_ok=True)
        out = {
            "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count_total": total_attempted,
            "count_results": total_processed,
            "failed_count": max(0, total_attempted - total_processed),
            "errors": all_errors,
            "top": []
        }
        with open("public/top_picks.json", "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print("[finalize] Wrote empty public/top_picks.json")
        return

    # Prepare matrices
    tickers = []
    numeric_matrix = {f: [] for f in NUMERIC_FEATURES}
    bool_matrix = {f: [] for f in BOOL_FEATURES}
    extras = []

    for r in rows:
        tickers.append(r.get('ticker'))
        for f in NUMERIC_FEATURES:
            numeric_matrix[f].append(float(r.get(f, 0.0)))
        for b in BOOL_FEATURES:
            bool_matrix[b].append(int(r.get(b, 0)))
        extras.append({'avg_vol20': float(r.get('avg_vol20', 0.0)), 'last_close': float(r.get('last_close', 0.0)), 'rsi': float(r.get('rsi',50.0))})

    # normalize numeric features
    norm_numeric = {}
    for f in NUMERIC_FEATURES:
        norm_numeric[f] = min_max_scale(numeric_matrix[f])

    # invert ATR (lower volatility better)
    if 'atr' in norm_numeric:
        norm_numeric['atr'] = 1.0 - norm_numeric['atr']

    # Build X matrix
    X_rows = []
    for i in range(len(tickers)):
        row = [float(norm_numeric[f][i]) for f in NUMERIC_FEATURES]
        row += [float(bool_matrix[b][i]) for b in BOOL_FEATURES]
        X_rows.append(row)
    X = np.array(X_rows, dtype=float)

    weights = np.array([WEIGHTS[f] for f in ALL_FEATURES], dtype=float)
    raw_scores = X.dot(weights)
    denom = np.sum(np.abs(weights)) if np.sum(np.abs(weights))!=0 else 1.0
    raw_scores = raw_scores / denom

    # normalize composite to 0..1
    minr = float(np.nanmin(raw_scores)); maxr = float(np.nanmax(raw_scores))
    if math.isclose(maxr, minr):
        norm_scores = np.full_like(raw_scores, 0.5)
    else:
        norm_scores = (raw_scores - minr) / (maxr - minr)

    scores_0_100 = (norm_scores * 100).round(2)
    scores_0_10 = (norm_scores * 10).round(2)

    # Build output items and sort
    items = []
    for i, tk in enumerate(tickers):
        numeric = {f: float(numeric_matrix[f][i]) for f in NUMERIC_FEATURES}
        boolean = {b: int(bool_matrix[b][i]) for b in BOOL_FEATURES}
        features = {**numeric, **boolean}
        expl = build_explanation({**features, 'rsi': extras[i]['rsi']})
        items.append({
            "ticker": tk,
            "score_0_100": float(scores_0_100[i]),
            "score_0_10": float(scores_0_10[i]),
            "features": features,
            "explanation": expl,
            "avg_vol20": extras[i]['avg_vol20'],
            "last_close": extras[i]['last_close']
        })

    # Sort by score, then macd_hist, then avg_vol20
    items.sort(key=lambda r: (r['score_0_100'], r['features'].get('macd_hist',0), r.get('avg_vol20',0)), reverse=True)

    out = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count_total": total_attempted,
        "count_results": total_processed,
        "failed_count": max(0, total_attempted - total_processed),
        "errors": all_errors,
        "top": items[:200]
    }

    os.makedirs("public", exist_ok=True)
    with open("public/top_picks.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print(f"[finalize] Wrote public/top_picks.json with top {len(out['top'])} items. attempted={total_attempted} processed={total_processed} failed={out['failed_count']} errors={len(all_errors)}")

if __name__ == "__main__":
    main()
