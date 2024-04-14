import requests
from bs4 import BeautifulSoup

from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

st = time.time()

def scroll(driver, timeout):
    scroll_pause_time = timeout

    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(scroll_pause_time)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            # If heights are the same it will exit the function
            break
        last_height = new_height

def scrape_yahoo_finance_news(ticker, num_news=5):
    """
    Scrape news headlines and links for a given ticker on Yahoo Finance using Selenium.

    Args:
    ticker (str): The ticker symbol of the stock.
    num_news (int): Number of news articles to scrape. Default is 5.

    Returns:
    list: A list of dictionaries containing 'title' and 'link' keys for each news article.
    """
    url = f"https://finance.yahoo.com/quote/{ticker}/latest-news"
    
    # Launch the web browser
    driver = webdriver.Chrome()  # You may need to specify the path to your chromedriver
    driver.get(url)
    
    news_list = []
    while len(news_list) < num_news:
        articles = driver.find_elements(By.CSS_SELECTOR, "li.stream-item.svelte-7rcxn:not(.adItem)")

        for article in articles:
            headline = article.find_element(By.CSS_SELECTOR,"h3.clamp.svelte-13zydns").text
            link = article.find_element(By.CSS_SELECTOR, "a.subtle-link.fin-size-small.titles.noUnderline.svelte-wdkn18").get_attribute('href')
            news_list.append({'title': headline, 'link': link})
        scroll(driver, 4)
    
    # Close the browser
    driver.quit()
    
    return news_list

# Example usage:
ticker = 'AAPL'
news_data = scrape_yahoo_finance_news(ticker, num_news=1000)

end = time.time()
print(end - st)
print(len(news_data))
for news in news_data:
    print(news)
