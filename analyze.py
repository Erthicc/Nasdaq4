# analyze.py
import pandas as pd, numpy as np, yfinance as yf
import pandas_ta as ta
from datetime import datetime

# Load ticker list
tickers = []
with open("nasdaqlisted.txt") as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts)>1 and parts[1] != 'Symbol' and not parts[0].startswith('File'):
            tickers.append(parts[1])
# (Optional: also load otherlisted.txt if desired)

results = []
for ticker in tickers:
    try:
        df = yf.download(ticker, period="14mo", interval="1d", progress=False)
        if df.empty: raise Exception("No data")
        df.rename(columns=str.capitalize, inplace=True)
        # Compute indicators (as above)...
        # -- MACD
        macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd']    = macd.macd()
        df['signal']  = macd.macd_signal()
        df['macd_hist']= macd.macd_diff()
        df['macd_slope'] = (df['macd_hist'].iloc[-1] - df['macd_hist'].iloc[-15]) / 14
        recent = df[['macd','signal']].tail(8)
        bullish = (recent['macd'] > recent['signal']).any() and (df['macd'].iloc[-1] > df['signal'].iloc[-1])
        df['macd_bull'] = bullish

        # RSI
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        df['rsi_slope'] = (df['rsi'].iloc[-1] - df['rsi'].iloc[-15]) / 14

        # SMA/EMA
        df['ema50'] = ta.trend.ema_indicator(df['Close'], window=50)
        df['sma20'] = ta.trend.sma_indicator(df['Close'], window=20)
        above = (df['Close'].iloc[-1] > df['ema50'].iloc[-1]) and (df['Close'].iloc[-1] > df['sma20'].iloc[-1])
        df['above_trend'] = int(above)

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        breakout = df['Close'].iloc[-1] > df['bb_upper'].iloc[-1]
        df['bb_breakout'] = int(breakout)

        # ADX
        df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)

        # ATR (volatility)
        df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)

        # OBV
        obv = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume'])
        df['obv'] = obv.on_balance_volume()
        df['obv_slope'] = (df['obv'].iloc[-1] - df['obv'].iloc[-15]) / 14

        # Volume spike
        df['vol20'] = df['Volume'].rolling(20).mean()
        vol_spike = df['Volume'].iloc[-1] > 1.5 * df['vol20'].iloc[-1]
        df['vol_spike'] = int(vol_spike)

        # Elliot-style wave (peak vs sma20)
        df['sma20'] = ta.trend.sma_indicator(df['Close'], window=20)
        recent = df['Close'].tail(60)
        peaks = recent[(recent.shift(1) < recent) & (recent.shift(-1) < recent)]
        if len(peaks)>0:
            last_peak = peaks.iloc[-1]
            wave_str = last_peak / df['sma20'].iloc[-1]
        else:
            wave_str = 1.0
        df['wave_strength'] = wave_str

        # Collect latest features
        feat = {
            'ticker': ticker,
            'macd_hist': df['macd_hist'].iloc[-1],
            'macd_slope': df['macd_slope'].iloc[-1],
            'macd_bull': int(df['macd_bull'].iloc[-1]),
            'rsi': df['rsi'].iloc[-1],
            'rsi_slope': df['rsi_slope'].iloc[-1],
            'wave_strength': df['wave_strength'].iloc[-1],
            'bb_breakout': df['bb_breakout'].iloc[-1],
            'adx': df['adx'].iloc[-1],
            'atr': df['atr'].iloc[-1],
            'obv_slope': df['obv_slope'].iloc[-1],
            'vol_spike': df['vol_spike'].iloc[-1],
            'above_trend': df['above_trend'].iloc[-1],
            'avg_vol20': df['vol20'].iloc[-1]
        }
        results.append(feat)
    except Exception as e:
        # skip ticker on error
        continue

# Create DataFrame of results
res_df = pd.DataFrame(results)
# Keep track of counts for JSON
count_total = len(tickers)
count_results = len(res_df)
failed_count = count_total - count_results

# Normalization (min-max)
from sklearn.preprocessing import MinMaxScaler
features = res_df[['macd_hist','macd_slope','rsi','rsi_slope',
                   'wave_strength','adx','atr','obv_slope']]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)
# Invert ATR (lower is better)
scaled[:, features.columns.get_loc('atr')] = 1 - scaled[:, features.columns.get_loc('atr')]

# Add boolean features (already 0/1) without scaling
res_df['macd_bull'] = res_df['macd_bull'].astype(int)
res_df['bb_breakout'] = res_df['bb_breakout'].astype(int)
res_df['vol_spike'] = res_df['vol_spike'].astype(int)
res_df['above_trend'] = res_df['above_trend'].astype(int)
bool_feats = res_df[['macd_bull','bb_breakout','vol_spike','above_trend']].values

# Combine scaled numeric and booleans
X = np.hstack([scaled, bool_feats])
feature_names = ['macd_hist','macd_slope','rsi','rsi_slope',
                 'wave_strength','adx','atr','obv_slope',
                 'macd_bull','bb_breakout','vol_spike','above_trend']

# Define weights
weights = np.array([2.0, 2.0, 1.5, 1.5, 2.0, 1.0, 1.0, 1.0, 3.0, 1.0, 0.5, 1.0])
raw_scores = X.dot(weights)
raw_scores /= np.sum(np.abs(weights))
res_df['score_norm'] = raw_scores
# Scale to 0-100
min_s, max_s = raw_scores.min(), raw_scores.max()
res_df['score_0_100'] = ((raw_scores - min_s)/(max_s-min_s) * 100).round(2)
res_df['score_0_10']  = (res_df['score_0_100'] / 10).round(2)

# Sort by score, then macd_hist, then avg_vol20
res_df.sort_values(by=['score_norm','macd_hist','avg_vol20'], ascending=False, inplace=True)

# Prepare output
top_n = 50
top_stocks = []
for _, row in res_df.head(top_n).iterrows():
    expl = []
    if row['macd_bull']: expl.append("recent MACD bullish crossover")
    if row['bb_breakout']: expl.append("Bollinger upper-band breakout")
    if row['adx'] > 25:  expl.append("strong trend (ADX {:.1f})".format(row['adx']))
    if row['vol_spike']: expl.append("volume spike")
    if row['above_trend']: expl.append("price above 50-day EMA & 20-day SMA")
    if row['rsi'] > 70: expl.append("RSI overbought")
    elif row['rsi'] < 30: expl.append("RSI oversold")
    if row['obv_slope'] > 0: expl.append("rising OBV")
    if row['wave_strength'] > 1.05: expl.append("strong upward wave")
    explanation = "; ".join(expl) if expl else "no major signal"
    top_stocks.append({
        "ticker": row['ticker'],
        "score_0_100": row['score_0_100'],
        "score_0_10": row['score_0_10'],
        "features": {f: float(row[f]) for f in ['macd_hist','macd_slope','rsi','rsi_slope','wave_strength','adx','atr','obv_slope','macd_bull','bb_breakout','vol_spike','above_trend']},
        "explanation": explanation
    })

output = {
    "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "count_total": count_total,
    "count_results": count_results,
    "failed_count": failed_count,
    "top": top_stocks
}
import json
with open("public/top_picks.json", "w") as f:
    json.dump(output, f, indent=2)
