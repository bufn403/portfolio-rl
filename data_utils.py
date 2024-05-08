import numpy as np
import pandas as pd


def exp_ffill(arr: pd.Series, gamma: float) -> pd.Series:
  assert 0 < gamma <= 1, f"{gamma=} is invalid; must be in (0, 1]"
  groups = pd.notna(arr).cumsum()
  exp = arr.isna().groupby(groups).cumsum()
  return arr.ffill().mul(gamma ** exp).fillna(0.0)


def read_crsp_data() -> pd.DataFrame:
  # read from csv
  df = pd.read_csv('crsp_snp100_2010_to_2024.csv', dtype='string')

  # convert datatypes for CRSP data
  df = df[['date', 'TICKER', 'PRC', 'VOL', 'ASKHI', 'BIDLO', 'FACPR']]
  df.date = pd.to_datetime(df.date)
  df.FACPR = df.FACPR.fillna('0.0')
  df.astype({
      'PRC': float,
      'VOL': float,
      'ASKHI': float,
      'BIDLO': float,
      'FACPR': float
  })
  df = df.drop_duplicates(subset=['date', 'TICKER'])
  df.dropna(inplace=True)

  return df


def read_sec_data(date_range: pd.DatetimeIndex, gamma: float = 1.0) -> pd.DataFrame:
  # read data from csv
  sec_df = pd.read_csv('sec_sentiment.csv', dtype='string')

  # clean sentiment data
  del sec_df['Unnamed: 0']
  sec_df['fdate'] = pd.to_datetime(sec_df['fdate'])
  sec_df.Sentiment = sec_df.Sentiment.astype(float)
  sec_df.lm_negative = sec_df.lm_negative.astype(float)
  sec_df.lm_positive = sec_df.lm_positive.astype(float)
  sec_df.lm_uncertainty = sec_df.lm_uncertainty.astype(float)
  sec_df.lm_negative = sec_df.groupby('TICKERH')['lm_negative'].transform(lambda v: exp_ffill(v, gamma))
  sec_df.lm_positive = sec_df.groupby('TICKERH')['lm_positive'].transform(lambda v: exp_ffill(v, gamma))
  sec_df.lm_uncertainty = sec_df.groupby('TICKERH')['lm_uncertainty'].transform(lambda v: exp_ffill(v, gamma))


  # fill in missing dates for sentiment
  min_date = min(sec_df.fdate.min(), date_range.min())
  max_date = max(sec_df.fdate.max(), date_range.max())
  date_range = pd.date_range(min_date, max_date, freq='D')
  full_df = pd.DataFrame({'fdate': list(date_range)}).merge(pd.DataFrame({'TICKERH': list(sec_df.TICKERH.unique())}), how='cross')
  full_df['neg'] = np.nan
  full_df['pos'] = np.nan
  full_df['unc'] = np.nan
  for ticker in sec_df.TICKERH.unique():
      ticker_df = sec_df[sec_df.TICKERH == ticker]
      ticker_df.index = pd.DatetimeIndex(ticker_df.fdate)
      ticker_df = ticker_df.reindex(date_range, fill_value=np.nan)
      full_df.loc[full_df.TICKERH == ticker, 'neg'] = ticker_df.lm_negative.ffill().values
      full_df.loc[full_df.TICKERH == ticker, 'pos'] = ticker_df.lm_positive.ffill().values
      full_df.loc[full_df.TICKERH == ticker, 'unc'] = ticker_df.lm_uncertainty.ffill().values
  
  return full_df


def read_news_data(date_range: pd.DatetimeIndex, gamma: float = 1.0) -> pd.DataFrame:
  # read data from csv
  news_df = pd.read_csv('./news_sentiment_data.csv', dtype='string')

  # clean sentiment data
  del news_df['Unnamed: 0']
  news_df.Date = pd.to_datetime(news_df.Date)
  news_df = news_df.sort_values(by = ['Ticker', 'Date'])
  news_df.Tone = news_df.Tone.astype(float)
  news_df.sentiment_embedding = news_df.sentiment_embedding.astype(float)

  min_date = max(news_df.Date.min(), date_range.min())
  max_date = min(news_df.Date.max(), date_range.max())
  date_range = pd.date_range(min_date, max_date, freq='D')
  full_df = pd.DataFrame({'date': list(date_range)}).merge(pd.DataFrame({'ticker': list(news_df.Ticker.unique())}), how='cross')
  full_df['sentiment_embedding'] = np.nan
  full_df['tone'] = np.nan

  for ticker in news_df.Ticker.unique():
      ticker_df = news_df[news_df.Ticker == ticker]
      ticker_df.index = pd.DatetimeIndex(ticker_df.Date)
      ticker_df = ticker_df.reindex(date_range, fill_value = np.nan)
      ticker_df['Ticker'] = ticker
      ticker_df.Tone = ticker_df.groupby('Ticker').Tone.transform(lambda v: exp_ffill(v, gamma))
      ticker_df.sentiment_embedding = ticker_df.groupby('Ticker').sentiment_embedding.transform(lambda v: exp_ffill(v, gamma))
      full_df.loc[full_df.ticker == ticker, 'tone'] = ticker_df.Tone.values
      full_df.loc[full_df.ticker == ticker, 'sentiment_embedding'] = ticker_df.sentiment_embedding.values

  full_df.sentiment_embedding = full_df.sentiment_embedding.fillna(0.0)
  full_df.tone = full_df.tone.fillna(0.0)

  return full_df


