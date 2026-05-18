import ccxt
import pandas as pd
import time
from datetime import datetime

def fetch_full_history(symbol='BTC/USD', time_frame='1h', start_date='2015-01-01T00:00:00Z', end_date='2026-02-27T23:59:59Z'):
    # Gunakan coinbaseexchange untuk data yang lebih dalam
    exchange = ccxt.coinbaseexchange()
    
    since = exchange.parse8601(start_date)
    end_timestamp = exchange.parse8601(end_date)
    
    all_ohlcv = []
    limit = 300 

    print(f"Mulai mengambil data {symbol} ({time_frame}) dari {start_date}...")

    while since < end_timestamp:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, time_frame, since, limit=limit)
            
            if not ohlcv or len(ohlcv) == 0:
                # Jika kosong, lompat 1 hari ke depan agar tidak stuck di masa lalu yang gak ada datanya
                since += 24 * 60 * 60 * 1000 
                # Cek agar tidak melompat melebihi waktu sekarang
                if since > exchange.milliseconds():
                    break
                continue
            
            all_ohlcv.extend(ohlcv)
            
            # Update 'since' ke timestamp candle terakhir + 1 milidetik
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1
            
            # Print progress agar kamu tahu program tidak macet
            print(f"Syncing: {exchange.iso8601(last_timestamp)} | Total rows: {len(all_ohlcv)}")
            
            time.sleep(exchange.rateLimit / 1000)

        except Exception as e:
            print(f"Error: {e}, mencoba lagi...")
            time.sleep(5)
            continue

    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Filter agar data tidak lewat dari end_date yang diminta
    df = df[df.index <= pd.to_datetime(end_timestamp, unit='ms')]

    return df

if __name__ == "__main__":
    df = fetch_full_history(time_frame='1h',end_date='2026-02-27T23:59:59Z')
    print(df.tail())

    df.to_csv('../data/btc_usd_ohlcv_1h.csv')