# NASDAQ Daily Technical Scanner

**What this repo does**

This project automatically scans the entire NASDAQ list daily, computes a broad set of investment-grade technical indicators for each ticker, ranks tickers by a configurable composite score, and publishes the results to a small GitHub Pages dashboard (`public/`). It is designed to run fully on GitHub using free tools and free public data (Yahoo Finance via `yfinance`).

---

## Quick highlights

- Pulls NASDAQ tickers from NasdaqTrader FTP (`nasdaqlisted.txt`).
- Downloads ~14 months of daily OHLCV via `yfinance`.
- Computes many indicators: MACD (histogram, slope, crossover), RSI (value & slope), Bollinger Bands breakout, ADX, ATR, OBV slope, volume spike vs 20-day average, EMA50/SMA20 alignment, and a simple Elliott-style wave strength heuristic.
- Normalizes features across the universe (min-max), inverts ATR (lower volatility preferred), applies configurable weights, computes a composite score, and maps to `0–100` and `0–10`.
- Generates human-readable explanations for top picks.
- Runs daily after market close via GitHub Actions and publishes `public/top_picks.json` + a dashboard to GitHub Pages.


---

## Not Financial Advice (NFA) — IMPORTANT

**This project is for educational and research purposes only. It is _not_ financial advice.**  
Use results at your own risk. Always do your own due diligence before making investment decisions. Past performance is not indicative of future results. The author/maintainer(Eric Bartelt) assume no responsibility for trading losses or other financial outcomes.

---

## No license included

This repository intentionally **does not** include a license file. Without a license, the default is “All rights reserved” — others can view the code but do not have the legal right to reuse, modify or distribute it.

---

## Files in this repo

- `download_nasdaq_list.py` — downloads the NASDAQ symbol list from NasdaqTrader FTP.
- `worker.py` — worker script that processes a chunk of tickers, writes `raw-results-{index}.json`.
- `finalize.py` — aggregator that merges worker outputs, normalizes, scores, builds explanations and writes `public/top_picks.json` and a simple dashboard if necessary.
- `.github/workflows/daily_scan.yml` — GitHub Actions workflow: parallel workers → aggregator → deploy to GitHub Pages.
- `public/` — static site files (`index.html`, `style.css`) and generated `top_picks.json`.
- `requirements.txt` — Python dependencies.
- `.gitignore`, `README.md` (this file).

---

