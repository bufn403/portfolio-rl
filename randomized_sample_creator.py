import pandas as pd
df = pd.read_csv('crsp_full_2010_2024.csv', dtype='string')
    
# convert datatypes
df = df[['date', 'TICKER', 'PRC', 'VOL', 'ASKHI', 'BIDLO', 'FACPR', 'vwretd']]
df.date = pd.to_datetime(df.date)
df.FACPR = df.FACPR.fillna('0.0')
df.astype({
    'PRC': float,
    'VOL': float,
    'ASKHI': float,
    'BIDLO': float,
    'FACPR': float,
    'vwretd': float
})

# drop duplicates and nans
df = df.drop_duplicates(subset=['date', 'TICKER'])
df.dropna(inplace=True)

# only include stocks that are present in all dates
ticker_ok = df.TICKER.value_counts() == df.TICKER.value_counts().max()
def is_max_val_count(ticker: str) -> bool:
  return ticker_ok[ticker]
ok = df.apply(lambda row: is_max_val_count(row['TICKER']), axis=1)
df = df[ok]
df = df[(df.date.dt.year >= 2010) & (df.date.dt.year < 2015)]

tickers = df['TICKER'].unique()

universe_sizes = [100, 200, 500, 1000]

for n in universe_sizes:
   df_reduced = df.loc[df['TICKER'].isin(pd.Series(tickers).sample(n = n, random_state = 1))]
   df_reduced.to_csv(f"{n}_random_sample_tickers.csv")
