import aiohttp
import asyncio
import pandas as pd
import numpy as np
import config

crsp = pd.read_csv('./crsp_snp100_2010_to_2024.csv')
tickers_list = crsp['TICKER'].unique()[:50]
ticker_query_str = ','.join(tickers_list)
start_date = '2015-01-01'
end_date = '2024-01-01'
url = 'https://api.benzinga.com/api/v2/news?token={}&tickers={}&dateFrom={}&dateTo={}&page={}'
site_data = []
pages = np.arange(1, 100, 1)

def get_tasks(session):
    tasks = [session.get(url.format(config.api_key, ticker_query_str, start_date, end_date, page)) for page in pages]
    return tasks

async def get_news_data():
    async with aiohttp.ClientSession() as session:
        tasks = get_tasks(session)
        responses = await asyncio.gather(*tasks)
        for response in responses:
            site_data.append(await response.text())

asyncio.run(get_news_data())
with open('site_xml_3.txt', 'w') as f:
    for item in site_data:
        f.write(f"{item}\n")
print(site_data)


    