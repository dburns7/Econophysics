#!/bin/python
# -----------------------------------------------------------------------------
#  File:        ScrapeData.py
#  Usage:       python ScrapeData.py 
#  Description: Scrape and clean 10-X documents from SEC EDGAR database, link to
#               historical stock returns from Yahoo Finance API
#  Created:     22-Feb-2016 Dustin Burns
# -----------------------------------------------------------------------------
import pandas as pd
from yahoo_finance import Share
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import wget
import nltk
import re
import os
from collections import Counter
from datetime import timedelta
from collections import OrderedDict
import random

def cik_to_ticker(cik):
  key = pd.read_csv('data/cik_ticker.csv', sep='|')  
  key.CIK = key.CIK.astype('str')
  try: 
    ind = key.CIK[key.CIK == cik].index.tolist()[0]
    ticker = key.Ticker[ind]
  except IndexError: 
    ticker = 'NaN'
  return ticker

def count_words(toks, dictionary):
  cnts = [0]*len(dictionary)
  for tok in toks:
    if tok.upper() in dictionary: cnts[dictionary.tolist().index(tok.upper())] += 1
  return cnts

if __name__ == "__main__":
  
  # Scrape and clean 10-X data 
  #data = pd.read_excel('data/LoughranMcDonald_10X_2014_test.xlsx', sheetname=0, skip_header=1)
  #data = pd.read_excel('data/test.xlsx', sheetname=0, skip_header=1)
  data = pd.read_excel('data/LoughranMcDonald_10X_2014.xlsx', sheetname=0, skip_header=1)
  N = 500
  data = data.loc[random.sample(list(data.index), N)]
  #data.to_csv(path_or_buf='data/LoughranMcDonald_10X_2014_test_2.xlsx')
  #data = data.head(2000)
  #data = pd.read_excel('data/LoughranMcDonald_10X_2014_test.xlsx', sheetname=0, skip_footer=910000)
  dictionary = pd.read_excel('data/LoughranMcDonald_MasterDictionary_2014.xlsx', sheetname=0)
  fin_neg = dictionary.Word[dictionary.Negative > 0].values
  print list(fin_neg)
  print 'wrongly' in fin_neg

  #print data.columns.values

  data = data.drop(['N_Positive', 'N_Uncertainty', 'N_Litigious', 'N_WeakModal', 'N_StrongModal', 'N_Constraining', 'N_Negation', 'GrossFileSize', 'NetFileSize', 'ASCIIEncodedChars', 'HTMLChars', 'XBRLChars', 'TableChars'], axis=1)
  data['N_Words_Paper'] = data.N_Words
  data['N_Unique_Words_Paper'] = data.N_Unique_Words
  data['N_Negative_Paper'] = data.N_Negative
  data = data.drop(['N_Negative'], axis=1)
  

  data.FILING_DATE  = data.FILING_DATE.astype('string')
  for time in data.FILING_DATE:
    t = time[0:4] + '-' + time[4:6] + '-' + time[6:8]
    data.ix[data.FILING_DATE == time, 'FILING_DATE'] = t
  data.FILING_DATE  = pd.to_datetime(data.FILING_DATE)
  
  #data['End_Date'] = data.FILING_DATE
  data['End_Date'] = data.FILING_DATE + timedelta(days=3)
  #print data
  '''
  delta = timedelta(days=1)
  for ind, date in enumerate(data.End_Date):
    i=0
    while i < 4:
      if date.weekday() not in [5, 6]:
        print data.End_Date[ind].weekday()
        data.End_Date[ind] = data.End_Date[ind] + delta
        i += 1
 
  #data.FILING_DATE  = data.FILING_DATE.astype('string')
  #data.End_Date  = data.End_Date.astype('string')
  print data
  ''' 
  #print data.head(100)
   
  toks_dirty_all = []
  toks_raw = []
  toks_total = []
  toks_unique = []
  toks_freq = []
  avg_freq = []
  neg_counts = []
  neg_total = []
  for name in data.FILE_NAME:
    fname = 'https://www.sec.gov/Archives/edgar/data/' + name.split('data_')[1].replace('_','/')
    
    data.ix[data.FILE_NAME == name, 'FILE_NAME'] = fname 
    try: 
      f = 'data/10X/' + fname.split('/')[-1]
      #if not os.path.exists(f):
      f = wget.download(fname, './data/10X')
    except:
      #data = data.drop(data.index[data.FILE_NAME == name])
      continue

    # uncleaned column
    text1 = open(f).read()
    toks_dirty = sorted(np.array(nltk.word_tokenize(text1)))
    toks_dirty_all.append(toks_dirty)

    # TO DO: compile regex before loop to speed up

    os.rename(f, f + '.orig')
    with open(f + '.orig', 'rb') as fin, open(f, 'wb') as fout:
      text = fin.read()
      text = re.sub(r'<IMS-HEADER>.*?</IMS-HEADER>', '', text, flags=re.DOTALL)
      text = re.sub(r'<SEC-HEADER>.*?</SEC-HEADER>', '', text, flags=re.DOTALL)
      text = re.sub(r'<GRAPHIC>.*?</GRAPHIC>', '', text, flags=re.DOTALL)
      text = re.sub(r'<ZIP>.*?</ZIP>', '', text, flags=re.DOTALL)
      text = re.sub(r'<EXCEL>.*?</EXCEL>', '', text, flags=re.DOTALL)
      text = re.sub(r'<PDF>.*?</PDF>', '', text, flags=re.DOTALL)
      text = re.sub(r'<XBRL>.*?</XBRL>', '', text, flags=re.DOTALL)
      text = re.sub(r'<.*?>', '', text, flags=re.DOTALL)
      fout.write(text)
    os.remove(f + '.orig')

    text = open(f).read()
    try: toks = np.array(nltk.word_tokenize(text))
    except: 
      #data = data.drop(data.index[data.FILE_NAME == name])
      continue
    
    # Clean words that contain numbers or special characters
    special_inds = [not bool(re.search('[\d\/\*\'\-,=;:@<>\.\_]', x)) for x in toks]  
    try: toks = toks[np.array(special_inds)]
    except: pass
    
    # Clean single char words
    word_inds = [len(x)>1 for x in toks]
    toks = toks[np.array(word_inds)]
   
    # Alphabetize
    #toks = np.concatenate((toks, ['ABANDON']))
    toks = sorted(toks) 

    toks_raw.append(toks)
    toks_total.append(len(toks))
    data.ix[data.FILE_NAME == fname, 'N_Words'] = len(toks) 
    #print 'N_words:        ' + str(len(toks))
    
    toks_dict = OrderedDict(sorted(Counter(toks).items()))
    toks_freq.append(toks_dict.values())
    avg_freq.append(np.nanmean(toks_dict.values()))
    toks_unique.append(toks_dict.keys())
    
    # add negation here for positive words
    # ['no', 'not', 'none', 'neither', 'never', 'nobody'] occurs four or fewer words before
    
    data.ix[data.FILE_NAME == fname, 'N_Unique_Words'] = len(toks_dict.keys()) 
    #print 'N_Unique_Words: ' + str(len(toks_dict.keys()))
    neg_in_dict = count_words(toks, fin_neg)
    neg_counts.append(neg_in_dict)
    neg_total.append(sum(neg_in_dict))
    #neg_counts.append(count_words(toks_dict.keys(), fin_neg))
 
  data['Toks_Dirty'] = toks_dirty_all
  data['Toks_Raw'] = toks_raw
  data['Toks_Total'] = toks_total
  data['Toks_Freq'] = toks_freq
  data['Avg_Freq'] = avg_freq
  data['Toks_Unique'] = toks_unique
  data['Neg_Counts'] = neg_counts
  data['Neg_Total'] = neg_total
  data['Prop_Yield'] = data.Neg_Total / data.Toks_Total
 
  ''' 
  test = np.array(['a', 'ab', '3-x', '--', '/HOME', '1', '3.4', '-as', 'ad', 'b', 'a1'])
  print test
  print sorted(test)
  
  #ind = [len(x)>1 for x in test]
  #test = test[np.array(ind)]
  #print 'singles removed'
  #print test
  
  ind = [not bool(re.search('[\d\/\*\'-]', x)) for x in test]  
  test = test[np.array(ind)]
  print 'numbers removed'
  print test
  
  #ind = []
  #for i, x in enumerate(test):
  #  try: 
  #    print (i, float(x))
  #    test = np.delete(test, i)
  #  except ValueError: continue
  #print test 
  '''
  
   
  # Remove companies not in key
  nan_ind = []
  tickers = []
  for ind, cik in enumerate(data.CIK): 
    ticker = cik_to_ticker(str(cik))
    if ticker == 'NaN': 
      nan_ind.append(ind)
      continue
    else: tickers.append(ticker)
  data = data.drop(data.index[nan_ind])
  data['Ticker'] = tickers
  data = data.reset_index(drop=True)
 
  # Get historical returns from yahoo_finance API 
  filing_return = []
  market_return = []
  for ind, ticker in enumerate(data.Ticker):
    share = Share(ticker)
    market = Share('^GSPC')
    #market = Share('^CRSPTM1')
    try: 
      returns = share.get_historical(str(data.FILING_DATE[ind])[0:10], str(data.End_Date[ind])[0:10])
      market = market.get_historical(str(data.FILING_DATE[ind])[0:10], str(data.End_Date[ind])[0:10])
    except: 
      filing_return.append(float('NaN'))
      market_return.append(float('NaN'))
      continue
    returns  = pd.DataFrame(returns)
    returns.Adj_Close = returns.Adj_Close.astype(float)
    filing_return.append( ((returns.Adj_Close.tail(1).values - returns.Adj_Close.head(1).values) / returns.Adj_Close.head(1).values)[0] )
    market = pd.DataFrame(market)
    market.Adj_Close = market.Adj_Close.astype(float)
    market_return.append( ((market.Adj_Close.tail(1).values - market.Adj_Close.head(1).values) / market.Adj_Close.head(1).values)[0])
  data['Filing_Return'] = filing_return
  data['Market_Return'] = market_return
  data['Excess_Return'] = data.Filing_Return - data.Market_Return

  data = data.dropna()
  data = data.reset_index(drop=True)
  
  # number of documents containing 1+ occurance of ith word
  df = [0] * len(dictionary)
  for cnts in data.Neg_Counts:
    for i, cnt in enumerate(cnts):
      if cnt > 0: df[i] += 1
  
  # Number of documents
  N = len(data.CIK.values)
  dict_len = len(fin_neg)
  
  # Calculate sum of weights, scaled counts
  w = [0] * N
  for j in range(0, N):
    weights = [0] * dict_len
    for i in range(0, dict_len):
      tf = data.Neg_Counts.values[j][i]
      if tf > 0:
        weights[i] = (1 + np.log(tf)) * np.log(N / df[i]) / (1 + np.log(data.Avg_Freq[j]))
    w[j] = sum(weights)
  data['Neg_Yield'] = w  
  
  print '---'
  print data.N_Negative_Paper
  print data.Neg_Total
  data.to_csv(path_or_buf='data/cleaned_10X_100.csv') 
   
