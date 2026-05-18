import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_hmm_backtest(df, regime_col='regime', return_col='log_return'):
    """
    Menjalankan simulasi backtest murni berdasarkan sinyal HMM.
    Regime 0 (Crash) = Jual/Cash. Regime 1 & 2 = Beli/Hold.
    """
    data = df.copy()
    
    # 1. Buat Sinyal Murni HMM
    # Jika regime == 0 (Crash), signal = 0 (Cash). Selain itu signal = 1 (Hold)
    data['signal'] = np.where(data[regime_col] == 0, 0, 1)
    
    # SHIFT 1 HARI (Sangat Penting untuk mencegah Look-Ahead Bias)
    # Kita bertransaksi HARI INI berdasarkan penutupan rezim KEMARIN
    data['signal'] = data['signal'].shift(1)
    
    # Hapus baris pertama yang menjadi NaN karena shift
    data.dropna(subset=['signal'], inplace=True)
    
    # 2. Hitung Return
    data['strategy_return'] = data[return_col] * data['signal']
    
    # Hitung Cumulative Return (Equity Curve)
    data['cum_bnh'] = data[return_col].cumsum().apply(np.exp)
    data['cum_strategy'] = data['strategy_return'].cumsum().apply(np.exp)
    
    # 3. Hitung Max Drawdown
    def calculate_max_drawdown(cum_returns):
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()
    
    mdd_bnh = calculate_max_drawdown(data['cum_bnh'])
    mdd_strategy = calculate_max_drawdown(data['cum_strategy'])
    
    # 4. Bungkus Statistik Akhir
    stats = {
        'Final_Value_BnH': data['cum_bnh'].iloc[-1],
        'Final_Value_Strategy': data['cum_strategy'].iloc[-1],
        'Max_Drawdown_BnH': mdd_bnh * 100,
        'Max_Drawdown_Strategy': mdd_strategy * 100
    }
    
    return data, stats