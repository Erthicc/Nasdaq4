# finalize.py
import json
import glob
import os
from datetime import datetime
import numpy as np
import math

# Feature list used for normalization and weighting
NUMERIC_FEATURES = ['macd_hist', 'macd_slope', 'rsi', 'rsi_slope',
                    'wave_strength', 'adx', 'atr', 'obv_slope']
BOOL_FEATURES = ['macd_bull', 'bb_breakout', 'vol_spike', 'above_trend']
ALL_FEATURES = NUMERIC_FEATURES + BOOL_FEATURES

# WEIGHTS (adjust to taste)
WEIGHTS = {
    'macd_hist': 2.0,
    'macd_slope': 2.0,
    'rsi': 1.5,
    'rsi_slope': 1.5,
    'wave_strength': 2.0,
    'adx': 1.0,
    'atr': 1.0,
    'obv_slope': 1.0,
    'macd_bull': 3.0,
    'bb_breakout': 1.0,
    'vol_spike': 0.5,
    'above_trend': 1.0
}

def load_raw_results(pattern="raw-results-*.json"):
    files = sorted(glob.glob(pattern))
    all_rows = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except Exception:
                data = []
            all_rows.extend(data)
    return all_rows

def min_max_scale(arr):
    arr = np.array(arr, dtype=float)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if math.isclose(mx, mn):
        # constant column: return 0.5 for everything
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
    rows = load_raw_results()
    if not rows:
        print("No raw result files found.")
        return

    # For stable ordering
    rows = sorted(rows, key=lambda r: r.get('ticker', ''))
    # Build matrices
    features_matrix = {f: [] for f in NUMERIC_FEATURES}
    bool_matrix = {f: [] for f in BOOL_FEATURES}
    tickers = []
    extras = []
    for r in rows:
        tickers.append(r.get('ticker'))
        for f in NUMERIC_FEATURES:
            v = r.get(f, 0.0)
            # convert nulls to 0
            try:
                features_matrix[f].append(float(v) if v is not None else 0.0)
            except Exception:
                features_matrix[f].append(0.0)
        for b in BOOL_FEATURES:
            bool_matrix[b].append(int(r.get(b, 0)))
        extras.append({'avg_vol20': float(r.get('avg_vol20', 0.0)), 'last_close': float(r.get('last_close', 0.0))})

    # Normalize numeric features min-max
    norm_numeric = {}
    for f in NUMERIC_FEATURES:
        arr = min_max_scale(features_matrix[f])
        norm_numeric[f] = arr

    # Invert ATR (lower volatility is better)
    if 'atr' in norm_numeric:
        norm_numeric['atr'] = 1.0 - norm_numeric['atr']

    # Create final normalized feature matrix
    X = []
    for i in range(len(tickers)):
        row = []
        for f in NUMERIC_FEATURES:
            row.append(float(norm_numeric[f][i]))
        for b in BOOL_FEATURES:
            row.append(int(bool_matrix[b][i]))
        X.append(row)
    X = np.array(X, dtype=float)

    # Build weights vector
    weights = np.array([WEIGHTS[f] for f in ALL_FEATURES])
    # compute raw score
    raw_scores = X.dot(weights)
    denom = np.sum(np.abs(weights))
    if denom == 0:
        denom = 1.0
    raw_scores = raw_scores / denom

    # normalize raw_scores to 0..1
    if raw_scores.size == 0:
        print("No scores to compute.")
        return
    minr = float(np.nanmin(raw_scores))
    maxr = float(np.nanmax(raw_scores))
    if math.isclose(maxr, minr):
        norm_scores = np.full_like(raw_scores, 0.5)
    else:
        norm_scores = (raw_scores - minr) / (maxr - minr)

    scores_0_100 = (norm_scores * 100).round(2)
    scores_0_10 = (norm_scores * 10).round(2)

    # Build output top list
    output_items = []
    for i, ticker in enumerate(tickers):
        feat_vals = {f: float(features_matrix[f][i]) for f in NUMERIC_FEATURES}
        bool_vals = {b: int(bool_matrix[b][i]) for b in BOOL_FEATURES}
        features = {**feat_vals, **bool_vals}
        expl = build_explanation({**{f: features[f] for f in features}, **extras[i], 'rsi': features['rsi']})
        output_items.append({
            'ticker': ticker,
            'score_0_100': float(scores_0_100[i]),
            'score_0_10': float(scores_0_10[i]),
            'features': features,
            'explanation': expl,
            'avg_vol20': extras[i]['avg_vol20'],
            'last_close': extras[i]['last_close']
        })

    # Sort by score, then macd_hist, then avg_vol20 (descending)
    output_items.sort(key=lambda r: (r['score_0_100'], r['features'].get('macd_hist', 0), r.get('avg_vol20', 0)), reverse=True)

    # Prepare public JSON
    out = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count_total": len(rows),
        "count_results": len(rows),
        "failed_count": 0,
        "top": output_items[:200]  # top 200 by default
    }

    os.makedirs("public", exist_ok=True)
    with open("public/top_picks.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Wrote public/top_picks.json")

    # Optionally make a simple index.html if not present
    if not os.path.exists("public/index.html"):
        html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Top NASDAQ Picks</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <h1>Top NASDAQ Picks</h1>
  <div id="meta"></div>
  <table id="table" border="1" cellpadding="6" cellspacing="0">
    <thead><tr><th>Rank</th><th>Ticker</th><th>Score</th><th>Signals</th></tr></thead>
    <tbody></tbody>
  </table>
  <script>
  fetch('top_picks.json').then(r=>r.json()).then(data=>{
    document.getElementById('meta').innerText = 'Generated at: ' + data.generated_at + ' • Total: ' + data.count_total;
    const tbody = document.querySelector('#table tbody');
    data.top.forEach((it,i)=> {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${i+1}</td><td>${it.ticker}</td><td>${it.score_0_100}</td><td>${it.explanation}</td>`;
      tbody.appendChild(tr);
    });
  });
  </script>
</body>
</html>"""
        with open("public/index.html", "w", encoding="utf-8") as fh:
            fh.write(html)
        print("Wrote public/index.html (auto-generated)")

if __name__ == "__main__":
    main()
