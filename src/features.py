import pandas as pd
import numpy as np

def add_features(df):
  data=df.copy()

  # add log returns
  data['log_returns']=np.log(data['close']/data['close'].shift(1))

  # add volatility
  data['volatility']=data['log_returns'].rolling(window=21).std()

  # trend slope (SMA)
  data['sma_50']=data['close'].rolling(window=50).mean()

  # the difference between today SMA and the SMA 5 days ago
  data['trend_slope']=data['sma_50'].diff(periods=5)

  # volume z-score
  vol_window=21
  epsilon=1e-8 # prevent dividing by 0
  vol_mean=data['volume'].rolling(window=vol_window).mean()
  vol_std=data['volume'].rolling(window=vol_window).std()
  data['volume_zscore']=(data['volume']-vol_mean)/(vol_std+epsilon)

  data.dropna(inplace=True)
  return data

def add_features_v2(df):
    data = df.copy()
    data['log_return'] = np.log(data['close'] / data['close'].shift(1))
    data['volatility'] = data['log_return'].rolling(window=21).std()

    sma_50 = data['close'].rolling(window=50).mean()
    data['trend_slope'] = (sma_50 - sma_50.shift(5)) / sma_50

    vol_mean = data['volume'].rolling(window=21).mean()
    vol_std = data['volume'].rolling(window=21).std()
    data['volume_zscore'] = (data['volume'] - vol_mean) / (vol_std + 1e-8)

    data['price_dist_ma'] = (data['close'] - sma_50) / sma_50
    data.dropna(inplace=True)

    return data

def add_features_v3(df):
    data = df.copy()

    # 1. Log Returns (Arah pergerakan harian)
    data['log_return'] = np.log(data['close'] / data['close'].shift(1))

    # 2. Volatility Fast (PERUBAHAN: 7 hari, bukan 21 hari)
    # Membuat model lebih sensitif terhadap perubahan mendadak
    data['volatility_fast'] = data['log_return'].rolling(window=7).std()

    # 3. Instant Volatility / Daily Range (FITUR BARU)
    # Seberapa liar harga berayun dalam satu hari penuh (High vs Low)
    # Ini mendeteksi kepanikan di hari H tanpa perlu menunggu besoknya
    data['daily_range'] = (data['high'] - data['low']) / data['open']

    # 4. Trend Slope (Persentase kemiringan tren jangka menengah)
    sma_50 = data['close'].rolling(window=50).mean()
    data['trend_slope'] = (sma_50 - sma_50.shift(5)) / sma_50

    # 5. Volume Z-Score (Partisipasi pasar)
    vol_mean = data['volume'].rolling(window=21).mean()
    vol_std = data['volume'].rolling(window=21).std()
    data['volume_zscore'] = (data['volume'] - vol_mean) / (vol_std + 1e-8)

    # 6. Relative Price Distance (Overbought / Oversold)
    data['price_dist_ma'] = (data['close'] - sma_50) / sma_50

    # Bersihkan baris yang mengandung NaN akibat perhitungan rolling
    data.dropna(inplace=True)
    
    return data


def add_aggressive_micro_features(df):
    """
    Ekstraksi fitur 1H spesifik untuk memburu anomali pergerakan harga.
    """
    data = df.copy()
    
    # 1. Log Returns 1H (The Baseline)
    data['log_ret_1h'] = np.log(data['close'] / data['close'].shift(1))
    
    # 2. Momentum Berjenjang (Rate of Change 3H, 6H, 12H)
    # XGBoost akan melihat kombinasi ini untuk mendeteksi akselerasi tren
    for h in [3, 6, 12]:
        data[f'roc_{h}h'] = np.log(data['close'] / data['close'].shift(h))
        
    # 3. Micro-Volatility (12H & 24H)
    # Volatilitas jangka pendek sering kali mendahului ledakan harga
    data['vol_12h'] = data['log_ret_1h'].rolling(window=12).std()
    data['vol_24h'] = data['log_ret_1h'].rolling(window=24).std()
    
    # 4. Volume Shocks (Z-Score Volume 24 Jam Terakhir)
    # Ini mendeteksi apakah volume saat ini adalah manipulasi "Paus" atau hanya noise ritel
    vol_mean = data['volume'].rolling(window=24).mean()
    vol_std = data['volume'].rolling(window=24).std()
    data['volume_shock'] = (data['volume'] - vol_mean) / (vol_std + 1e-8)
    
    # 5. Wick Intensity (Shadow Candlestick)
    # Ini krusial! Mengukur penolakan harga (Rejection).
    # Upper Wick panjang = Buyer kehabisan nafas. Lower Wick panjang = Seller panik.
    candle_range = data['high'] - data['low'] + 1e-8
    data['upper_wick_ratio'] = (data['high'] - np.maximum(data['close'], data['open'])) / candle_range
    data['lower_wick_ratio'] = (np.minimum(data['close'], data['open']) - data['low']) / candle_range
    
    # 6. Micro Trend (Distance from EMA & EMA Cross) # Exponential Moving Average lebih responsif terhadap perubahan harga terbaru dibanding SMA
    # Seberapa jauh harga menyimpang dari nilai rata-rata institusi jangka pendek
    ema_9 = data['close'].ewm(span=9, adjust=False).mean() #ewm=exponential weighted moving memberi bobot lebih besar di harga terbaru
    ema_21 = data['close'].ewm(span=21, adjust=False).mean()
    
    data['dist_ema9'] = (data['close'] - ema_9) / ema_9
    data['ema_cross_signal'] = (ema_9 - ema_21) / ema_21
    
    data.dropna(inplace=True)
    
    print(f"Ekstraksi selesai. Dihasilkan {len(data.columns) - 6} fitur baru.")
    return data