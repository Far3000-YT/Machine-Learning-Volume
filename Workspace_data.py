import ccxt
import pandas as pd
import time
import os

#using binance
exchange = ccxt.binance()

symbol = 'BTC/USDT' #using USDT for trading pair
timeframe = '1m' #1 minute bars
output_dir = 'data/binance/btc_usdt/1m' #output folder

#create folder if not exists
os.makedirs(output_dir, exist_ok=True)

def fetch_ohlcv(exchange, symbol, timeframe, since, limit=1000):
    #fetch data using ccxt
    #since is timestamp in milliseconds
    try:
        #rate limit handling
        time.sleep(exchange.rateLimit / 1000)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        return ohlcv
    except Exception as e:
        #print error if fetch fails
        print(f"Error fetching data: {e}")
        return None

start_time_ms = 1640995200000 #timestamp start (here from january 1st 2022 !) (might take space)

all_ohlcv = []
#fetch loop
#fetch in chunks because limit is 1000
while True:
    #fetch data chunk
    data = fetch_ohlcv(exchange, symbol, timeframe, start_time_ms)
    if data is None or len(data) == 0:
        #no more data or error, break loop
        print("No more data or error. Breaking.")
        break

    #extend list with fetched data
    all_ohlcv.extend(data)
    #update start time for next fetch
    start_time_ms = data[-1][0] + 1 #start from next millisecond

    #print progress
    print(f"Fetched {len(all_ohlcv)} data points...")

    time.sleep(0.05) #wait a bit


#convert list to dataframe
df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

#convert timestamp to datetime objects
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

#set timestamp as index
df.set_index('timestamp', inplace=True)

#handle duplicates if any
df = df[~df.index.duplicated(keep='first')]

#print info
print(df.info())

#define output file path
output_file = os.path.join(output_dir, f'{symbol.replace("/", "_")}_{timeframe}.parquet') #use parquet format

#save dataframe to parquet
df.to_parquet(output_file)

#confirm save
print(f"Data saved to {output_file}")
#merge pls