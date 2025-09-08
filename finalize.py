#!/usr/bin/env python3
# finalize.py
"""
Aggregate worker artifacts, compute scores, write public/top_picks.json,
and generate a minimal static dashboard in public/ (index.html, app.js, style.css)
so GitHub Pages will display the results.

This script intentionally avoids heavy numeric libraries to maximize
compatibility in the CI environment.
"""

import json
import glob
import os
import math
from datetime import datetime

# Which features to expect (must match worker output keys)
NUMERIC_FEATURES = ['macd_hist', 'macd_slope', 'rsi', 'rsi_slope',
                    'wave_strength', 'adx', 'atr', 'obv_slope']
BOOL_FEATURES = ['macd_bull', 'bb_breakout', 'vol_spike', 'above_trend']
ALL_FEATURES = NUMERIC_FEATURES + BOOL_FEATURES

# Weights controlling the composite score (tweak to taste)
WEIGHTS = {
    'macd_hist': 2.0, 'macd_slope': 2.0, 'rsi': 1.5, 'rsi_slope': 1.5,
    'wave_strength': 2.0, 'adx': 1.0, 'atr': 1.0, 'obv_slope': 1.0,
    'macd_bull': 3.0, 'bb_breakout': 1.0, 'vol_spike': 0.5, 'above_trend': 1.0
}

# Output files
PUBLIC_DIR = "public"
JSON_OUT = os.path.join(PUBLIC_DIR, "top_picks.json")
INDEX_OUT = os.path.join(PUBLIC_DIR, "index.html")
APP_JS_OUT = os.path.join(PUBLIC_DIR, "app.js")
STYLE_OUT = os.path.join(PUBLIC_DIR, "style.css")

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

def safe_float(x):
    try:
        if x is None:
            return 0.0
        return float(x)
    except Exception:
        return 0.0

def min_max_scale_list(values):
    # values: list of floats
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    if math.isclose(mx, mn):
        # all same -> return 0.5 for each element
        return [0.5 for _ in values]
    return [(v - mn) / (mx - mn) for v in values]

def build_explanation(row):
    expl = []
    if int(row.get('macd_bull', 0)): expl.append("recent MACD bullish crossover")
    if int(row.get('bb_breakout', 0)): expl.append("Bollinger upper-band breakout")
    try:
        adx = float(row.get('adx', 0))
        if adx > 25: expl.append(f"strong trend (ADX {adx:.1f})")
    except Exception:
        pass
    if int(row.get('vol_spike', 0)): expl.append("volume spike")
    if int(row.get('above_trend', 0)): expl.append("price above SMA/EMA trend")
    try:
        rsi = float(row.get('rsi', 50))
        if rsi > 70: expl.append("RSI overbought")
        elif rsi < 30: expl.append("RSI oversold")
    except Exception:
        pass
    try:
        if float(row.get('obv_slope', 0)) > 0:
            expl.append("rising OBV")
    except Exception:
        pass
    try:
        if float(row.get('wave_strength', 1)) > 1.05:
            expl.append("strong upward wave")
    except Exception:
        pass
    return "; ".join(expl) if expl else "no significant signals"

def aggregate_and_score():
    artifacts = load_worker_artifacts()
    total_attempted = 0
    total_processed = 0
    all_errors = []
    rows = []

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

    print(f"[finalize] loaded {len(artifacts)} artifact files; attempted_total={total_attempted} processed_total={total_processed} rows={len(rows)} errors={len(all_errors)}")

    if not rows:
        # still write an empty top_picks.json with metadata and create a minimal index.html
        os.makedirs(PUBLIC_DIR, exist_ok=True)
        out = {
            "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count_total": total_attempted,
            "count_results": total_processed,
            "failed_count": max(0, total_attempted - total_processed),
            "errors": all_errors,
            "top": []
        }
        with open(JSON_OUT, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print("[finalize] Wrote empty", JSON_OUT)
        write_dashboard_placeholder()
        return out

    # Build matrices (pure python)
    tickers = []
    numeric_values = {f: [] for f in NUMERIC_FEATURES}
    bool_values = {f: [] for f in BOOL_FEATURES}
    extras = []

    for r in rows:
        tickers.append(r.get('ticker'))
        for f in NUMERIC_FEATURES:
            numeric_values[f].append(safe_float(r.get(f, 0.0)))
        for b in BOOL_FEATURES:
            bool_values[b].append(1 if int(r.get(b, 0)) else 0)
        extras.append({
            'avg_vol20': safe_float(r.get('avg_vol20', 0.0)),
            'last_close': safe_float(r.get('last_close', 0.0)),
            'rsi': safe_float(r.get('rsi', 50.0))
        })

    # Normalize numeric features
    norm_numeric = {}
    for f in NUMERIC_FEATURES:
        norm_numeric[f] = min_max_scale_list(numeric_values[f])

    # invert ATR (lower is better)
    if 'atr' in norm_numeric:
        norm_numeric['atr'] = [1.0 - v for v in norm_numeric['atr']]

    # Build score vector
    items = []
    weights = [WEIGHTS[f] for f in ALL_FEATURES]
    abs_weight_sum = sum(abs(w) for w in weights) or 1.0

    for i, tk in enumerate(tickers):
        feature_vector = []
        for f in NUMERIC_FEATURES:
            feature_vector.append(norm_numeric[f][i])
        for b in BOOL_FEATURES:
            feature_vector.append(bool_values[b][i])

        raw_score = sum(v * w for v, w in zip(feature_vector, weights))
        # stabilize: divide by sum of absolute weights to map within some sensible range
        composite = raw_score / abs_weight_sum
        items.append({
            "ticker": tk,
            "raw_composite": composite,
            "features_numeric": {f: numeric_values[f][i] for f in NUMERIC_FEATURES},
            "features_bool": {b: bool_values[b][i] for b in BOOL_FEATURES},
            "extras": extras[i],
        })

    # normalize composite to 0..1
    comps = [it['raw_composite'] for it in items]
    mn = min(comps)
    mx = max(comps)
    if math.isclose(mx, mn):
        for it in items:
            it['score_01'] = 0.5
    else:
        for it in items:
            it['score_01'] = (it['raw_composite'] - mn) / (mx - mn)

    # scale to 0..100 and 0..10
    for it in items:
        it['score_0_100'] = round(it['score_01'] * 100, 2)
        it['score_0_10'] = round(it['score_01'] * 10, 2)
        # human explanation
        merged_features = {}
        merged_features.update(it['features_numeric'])
        merged_features.update(it['features_bool'])
        merged_features['rsi'] = it['extras'].get('rsi', 50.0)
        it['explanation'] = build_explanation(merged_features)

    # Sort by score desc, tiebreaker: macd_hist then avg_vol20
    def sort_key(it):
        macd_hist = it['features_numeric'].get('macd_hist', 0)
        vol = it['extras'].get('avg_vol20', 0)
        return (it['score_0_100'], macd_hist, vol)

    items.sort(key=sort_key, reverse=True)

    # limit top results (you can change)
    top_items = []
    for it in items[:500]:
        top_items.append({
            "ticker": it['ticker'],
            "score_0_100": it['score_0_100'],
            "score_0_10": it['score_0_10'],
            "features": {**it['features_numeric'], **it['features_bool']},
            "explanation": it['explanation'],
            "avg_vol20": it['extras'].get('avg_vol20'),
            "last_close": it['extras'].get('last_close')
        })

    out = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count_total": total_attempted,
        "count_results": total_processed,
        "failed_count": max(0, total_attempted - total_processed),
        "errors": all_errors,
        "top": top_items
    }

    # ensure public dir
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    with open(JSON_OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"[finalize] Wrote {JSON_OUT} with top {len(top_items)} items")

    # write dashboard files
    write_dashboard_files()
    return out

def write_dashboard_placeholder():
    # minimal placeholder index if there are no results
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NASDAQ Scan — no results</title>
</head>
<body>
<h2>NASDAQ Scan — No Results</h2>
<p>The analysis ran but produced no results. Check the workflow logs and artifacts.</p>
<p>Generated at: {ts}</p>
</body>
</html>""".format(ts=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
    with open(INDEX_OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    # write a tiny CSS too
    css = "body { font-family: Arial, sans-serif; padding: 20px; }"
    with open(STYLE_OUT, "w", encoding="utf-8") as fh:
        fh.write(css)

def write_dashboard_files():
    # index.html - simple app shell that loads app.js
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>NASDAQ Daily Scan</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <div class="container">
    <header>
      <h1>NASDAQ Daily Scan</h1>
      <div id="meta"></div>
    </header>

    <main>
      <div id="controls">
        <input id="search" placeholder="Filter tickers (symbol or explanation)" />
        <label>Top: <input id="topN" type="number" value="50" min="1" max="500" /></label>
      </div>

      <div id="summary"></div>

      <table id="results">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Ticker</th>
            <th>Score (0-100)</th>
            <th>Last Close</th>
            <th>Avg Vol(20)</th>
            <th>Explanation</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>

      <footer>
        <p>Generated at <span id="generated_at"></span>. JSON: <a href="top_picks.json">top_picks.json</a></p>
      </footer>
    </main>
  </div>
  <script src="app.js"></script>
</body>
</html>
"""
    with open(INDEX_OUT, "w", encoding="utf-8") as fh:
        fh.write(html)

    # style.css - minimal styling
    css = """
:root{--bg:#f7f8fb;--card:#fff;--accent:#0b5fff;--muted:#666}
body{font-family:Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; background:var(--bg); color:#111; margin:0}
.container{max-width:1100px;margin:28px auto;padding:18px}
header h1{margin:0 0 6px 0}
#controls{margin:12px 0 18px 0;display:flex;gap:10px;align-items:center}
#search{padding:8px 10px;border-radius:6px;border:1px solid #ddd}
#topN{width:80px;padding:6px}
table{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden}
thead th{background:#fafafa;text-align:left;padding:12px;border-bottom:1px solid #eee}
tbody td{padding:10px;border-bottom:1px solid #f1f1f1}
tbody tr:hover{background:#fcfdff}
footer{margin-top:14px;color:var(--muted)}
"""
    with open(STYLE_OUT, "w", encoding="utf-8") as fh:
        fh.write(css)

    # app.js - fetches top_picks.json and displays table + simple filtering
    js = r"""
(async function(){
  const metaEl = document.getElementById('meta');
  const genEl = document.getElementById('generated_at');
  const summaryEl = document.getElementById('summary');
  const tbody = document.querySelector('#results tbody');
  const search = document.getElementById('search');
  const topN = document.getElementById('topN');

  async function loadData(){
    try{
      const res = await fetch('top_picks.json', {cache:'no-store'});
      if(!res.ok) { throw new Error('Fetch error '+res.status); }
      const j = await res.json();
      metaEl.textContent = `Scores: ${j.count_results || 0} results (attempted ${j.count_total || 0})`;
      genEl.textContent = j.generated_at || '';
      render(j.top || []);
    }catch(err){
      metaEl.textContent = 'Error loading top_picks.json: '+err;
      console.error(err);
    }
  }

  function render(items){
    const limit = Math.max(1, Math.min(500, parseInt(topN.value||50)));
    const q = (search.value||'').toLowerCase().trim();
    let shown = 0;
    tbody.innerHTML = '';
    for(let i=0;i<items.length && shown<limit;i++){
      const it = items[i];
      const txt = (it.ticker+' '+(it.explanation||'')).toLowerCase();
      if(q && !txt.includes(q)) continue;
      const tr = document.createElement('tr');
      const rankTd = document.createElement('td'); rankTd.textContent = (i+1);
      const tickerTd = document.createElement('td'); tickerTd.textContent = it.ticker;
      const scoreTd = document.createElement('td'); scoreTd.textContent = it.score_0_100;
      const lastTd = document.createElement('td'); lastTd.textContent = (it.last_close||'').toString();
      const volTd = document.createElement('td'); volTd.textContent = (it.avg_vol20||'').toString();
      const explTd = document.createElement('td'); explTd.textContent = it.explanation || '';
      tr.appendChild(rankTd); tr.appendChild(tickerTd); tr.appendChild(scoreTd); tr.appendChild(lastTd); tr.appendChild(volTd); tr.appendChild(explTd);
      tbody.appendChild(tr);
      shown++;
    }
    summaryEl.textContent = `Showing ${shown} of ${items.length} tickers`;
  }

  search.addEventListener('input', ()=> loadData());
  topN.addEventListener('change', ()=> loadData());

  await loadData();
})();
"""
    with open(APP_JS_OUT, "w", encoding="utf-8") as fh:
        fh.write(js)
    print("[finalize] Wrote index.html, app.js, style.css into", PUBLIC_DIR)

if __name__ == "__main__":
    aggregate_and_score()
