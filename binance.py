import ccxt 
import pprint 
import pandas as pd
import time

binance = ccxt.binance()
date = time.time()
pair = 'ETH'
# markets = binance.load_markets()
# print(markets.keys())
# print(len(markets)
btc_ohlcv = binance.fetch_ohlcv(f"{pair}/USDT")
columns=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
df = pd.DataFrame(btc_ohlcv, columns=columns)
# for i, col in enumerate(columns[1:]):
#     columns[i] = col.title()


# df['timestamp'] = pd.to_datetime(df['datetime'], unit='ms')
df['timestamp'] = (df['datetime']/1000).astype(int)
del df['datetime']
df.set_index('timestamp', inplace=True)
df.to_csv(f"{int(date)}_{pair}.csv")

print(df)