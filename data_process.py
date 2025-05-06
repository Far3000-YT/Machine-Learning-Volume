import pandas as pd
import numpy as np
import os

#load the parquet file
input_file = 'data/binance/btc_usdt/1m/BTC_USDT_1m.parquet'

#check if file exists
if not os.path.exists(input_file):
    #print error file not found
    print(f"Error: Data file not found at {input_file}. Please run fetch_data.py first.")
    exit() #exit if no file

#read parquet into dataframe
df = pd.read_parquet(input_file)

#print info
print("--- Original Data Info ---")
print(df.info())
print("\n--- Original Data Head ---")
print(df.head())


#plotting code here would show price and volume
#can use matplotlib, seaborn, or plotly
#example:
#import matplotlib.pyplot as plt
#df['close'].plot(figsize=(12, 6))
#df['volume'].plot(secondary_y=True, style='g')
#plt.show()
#we can run this section interactively in a notebook or terminal
#no need to keep plot code in final script maybe


#this is main part, create indicators
df['volume_sma_10'] = df['volume'].rolling(window=10).mean() #10 bar simple moving average volume
df['volume_ema_10'] = df['volume'].ewm(span=10, adjust=False).mean() #10 bar exponential moving average volume

df['volume_change'] = df['volume'].pct_change() #percent change volume

#relative volume: volume compared to average
#avoid division by zero if volume is zero
df['relative_volume_10'] = np.where(
    df['volume_sma_10'] > 0,
    df['volume'] / df['volume_sma_10'],
    0 #handle zero average volume
)

#volume spike indicator
#volume > N std dev above MA
window_vol_spike = 20
num_std_dev = 2
df['volume_ma_spike'] = df['volume'].rolling(window=window_vol_spike).mean()
df['volume_std_spike'] = df['volume'].rolling(window=window_vol_spike).std()
df['volume_spike'] = (
    df['volume'] > df['volume_ma_spike'] + num_std_dev * df['volume_std_spike']
).astype(int) #1 if spike, 0 otherwise

#price-volume interaction features
df['price_change'] = df['close'].diff() #price change current bar
df['price_change_vol'] = df['price_change'] * df['volume'] #price change scaled by volume

#candlestick body size scaled by volume
df['body_abs'] = abs(df['close'] - df['open'])
df['body_abs_vol'] = df['body_abs'] * df['volume']

#example using ta-lib or pandas-ta for more indicators
#install: pip install pandas_ta
#import pandas_ta as ta
#df.ta.obv(append=True)
#df.ta.ad(append=True)
#df.ta.cmf(append=True)


#lagged features: past values
#lag volume by 1 bar
df['volume_lag_1'] = df['volume'].shift(1)
#lag price change by 1 bar
df['price_change_lag_1'] = df['price_change'].shift(1)
#lag our relative volume feature
df['relative_volume_10_lag_1'] = df['relative_volume_10'].shift(1)

#add more lagged features as needed


#predict if price is higher in N minutes
future_window = 5 #predict 5 minutes ahead
#calculate future close price
df['future_close'] = df['close'].shift(-future_window)

#create binary target: 1 if future close > current close, 0 otherwise
#handle last N rows where future_close is NaN
df['target'] = (df['future_close'] > df['close']).astype(int)

#replace inf/-inf with NaN because StandardScaler cant handle them
#check whole dataframe for inf
df.replace([np.inf, -np.inf], np.nan, inplace=True)

#handle NaNs introduced by shifts and rolling windows
df.dropna(inplace=True)


#at this point, you'd analyze correlations, look at feature importance
#for now, we keep all engineered features
#if you added TA-lib/pandas_ta indicators, they'd be columns too


output_file_processed = 'data/processed/btc_usdt_1m_features.parquet' #new folder for processed data

#create processed data folder if not exists
os.makedirs(os.path.dirname(output_file_processed), exist_ok=True)

#save dataframe with features and target
df.to_parquet(output_file_processed)

print("\n--- Processed Data Info ---")
print(df.info())
print("\n--- Processed Data Head ---")
print(df.head())
print(f"\nProcessed data with features and target saved to {output_file_processed}")