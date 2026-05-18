import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score
from train_xgboost import Xgboost_Model

def run_walk_forward_validation(df, train_months=6, test_months=1):
    """
    Menjalankan simulasi Walk-Forward murni pada timeframe 1H.
    1 bulan diasumsikan = 30 hari = 720 jam.
    """
    print(f"Mempersiapkan mesin Walk-Forward. Train: {train_months} Bulan, Test: {test_months} Bulan.")
    
    # Asumsi timeframe 1 Jam
    hours_per_month = 24 * 30
    train_window = train_months * hours_per_month
    test_window = test_months * hours_per_month
    
    # 1. Karantina Fitur
    forbidden_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'future_return']
    features = [col for col in df.columns if col not in forbidden_cols]
    
    print(f"Bahan baku tervalidasi. Menggunakan {len(features)} fitur murni.")
    print(f"Fitur: {features}\n")
    
    total_rows = len(df)
    results = []
    
    # 2. THE LOOP (Iterasi Mesin Waktu)
    # Kita melangkah sejauh 'test_window' setiap iterasinya
    step = 1
    for start_idx in range(0, total_rows - train_window - test_window + 1, test_window):
        end_train = start_idx + train_window
        end_test = end_train + test_window
        
        # Potong DataFrame (Slicing)
        train_slice = df.iloc[start_idx:end_train]
        test_slice = df.iloc[end_train:end_test]
        
        test_start_date = test_slice.index[0].strftime('%Y-%m-%d')
        test_end_date = test_slice.index[-1].strftime('%Y-%m-%d')
        
        print(f"--- Step {step} | Trading Window: {test_start_date} to {test_end_date} ---")
        
        # 3. INSTANSIASI & COOKING (Model Baru Setiap Step)
        # Kita panggil class yang sudah kamu buat sebelumnya
        bot = Xgboost_Model(n_estimators=150, max_depth=4, learning_rate=0.05)
        bot.fit(train_slice, features, target_col='target', verbose=False)
        
        # 4. PREDIKSI MASA DEPAN (Out-of-Sample)
        y_true = test_slice['target'].values
        y_pred = bot.predict(test_slice)
        
        # 5. QUALITY CONTROL (Catat Hasil)
        # Rata-rata kemurnian tembakan Long (1) dan Short (-1)
        # zero_division=0 mencegah error jika bot tidak menembak sama sekali
        prec_long = precision_score(y_true, y_pred, labels=[1], average='macro', zero_division=0)
        prec_short = precision_score(y_true, y_pred, labels=[-1], average='macro', zero_division=0)
        
        # Hitung seberapa aktif bot kita (Berapa kali dia menembak dari total jam?)
        total_long_signals = np.sum(y_pred == 1)
        total_short_signals = np.sum(y_pred == -1)
        
        print(f"  > Long  Precision: {prec_long*100:.1f}% | Sinyal: {total_long_signals}")
        print(f"  > Short Precision: {prec_short*100:.1f}% | Sinyal: {total_short_signals}")
        
        # Simpan evaluasi ke buku besar (Ledger)
        results.append({
            'step': step,
            'start_date': test_start_date,
            'end_date': test_end_date,
            'precision_long': prec_long,
            'precision_short': prec_short,
            'signals_long': total_long_signals,
            'signals_short': total_short_signals,
            'top_feature': bot.feature_importance_['Feature'].iloc[0] # Lihat fitur apa yang memimpin di bulan ini
        })
        
        step += 1
        
    print("\nSimulasi Walk-Forward Selesai.")
    
    # Konversi hasil ke DataFrame agar mudah dianalisis
    df_results = pd.DataFrame(results)
    return df_results