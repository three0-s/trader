import ccxt 
import pprint 
import pandas as pd


binance = ccxt.binance()
pair = 'BTC'
# markets = binance.load_markets()
# print(markets.keys())
# print(len(markets)
btc_ohlcv = binance.fetch_ohlcv(f"{pair}/USDT")
columns=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
df = pd.DataFrame(btc_ohlcv, columns=columns)
# for i, col in enumerate(columns[1:]):
#     columns[i] = col.title()


# df['timestamp'] = pd.to_datetime(df['datetime'], unit='ms')
df['Timestamp'] = df['datetime']
del df['datetime']
df.set_index('Timestamp', inplace=True)
df.to_csv(f"230514{pair}.csv")

print(df)