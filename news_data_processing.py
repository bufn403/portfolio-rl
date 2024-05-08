import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import datetime
import gc
import logging
import time
from functools import wraps
import errno
import os
import signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pd.set_option('display.max_colwidth', None)

partner_headlines = pd.read_csv('./archive/raw_partner_headlines.csv')
benzinga_headlines = pd.read_csv('./archive/raw_analyst_ratings.csv')
headlines = pd.concat([partner_headlines, benzinga_headlines]).drop('Unnamed: 0', axis = 1)

sp_100_data = pd.read_html('https://en.wikipedia.org/wiki/S%26P_100')
sp_100_cmpys = sp_100_data[2]
sp_100 = sp_100_cmpys['Symbol'].to_list()
sp_100[]

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert", token = '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')
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", token = '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').to("mps")

class NewsHeadlines(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt').to("mps")
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


class TimeoutError(Exception):
    pass

def timeout(seconds=100, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator

@timeout(100)
def get_batch_logits(batch, model):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)[0]
        probabilities = torch.softmax(logits, dim=1)
    return probabilities

sp_100_full_hls = headlines.loc[headlines['stock'].isin(sp_100)]

dataset = NewsHeadlines(sp_100_full_hls['headline'].to_list(), tokenizer)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

# Perform inference in batches
i = 1
logging.info("Dataset Loaded")
for batch in dataloader:
    try:
        probabilities = get_batch_logits(batch, model)
        torch.save(probabilities, f"./BatchedTensors/tensor{i}.pt")
        logging.info(f"Batch {i} completed")
    except TimeoutError:
        logging.into(f"Batch {i} timed out")
    i += 1