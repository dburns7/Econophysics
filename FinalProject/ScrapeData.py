#!/bin/python
# -----------------------------------------------------------------------------
#  File:        StockDistributionFitting.py
#  Usage:       python StockDistributionFitting.py 
#  Description: Project to fit distributions of historical stock returns
#  Created:     22-Feb-2016 Dustin Burns
# -----------------------------------------------------------------------------
import pandas as pd
from yahoo_finance import Share
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import wget
import nltk
import re
import os

if __name__ == "__main__":

  # Parse command line arguments
  #parser = argparse.ArgumentParser(description="Compute and plot trailing returns")
  #parser.add_argument('--ticker', nargs='?', default='YHOO', help='Fund ticker symbol')
  #parser.add_argument('--start_date', nargs='?', default='2013-04-25')
  #parser.add_argument('--end_date', nargs='?', default='2017-01-10')
  #args = parser.parse_args()

  # Get data
  #share = Share(args.ticker)
  #data  = share.get_historical(args.start_date, args.end_date)
  #data  = pd.DataFrame(data)
  #data.Date  = pd.to_datetime(data.Date)
  #data.set_index(data.Date, inplace=True)
  #data.sort_index(inplace=True)
  #data.Adj_Close = data.Adj_Close.astype(float)
  #Close = data.T.values[0]
  
  
  # Read in a clean data 
  data = pd.read_excel('data/LoughranMcDonald_10X_2014_test.xlsx', sheetname=0, skip_header=1)
  #data = pd.read_excel('data/LoughranMcDonald_10X_2014_test.xlsx', sheetname=0, skip_footer=910000)
  print data.columns.values
  print data.head(100)
  names = data.FILE_NAME
  toks_total = []
  toks_unique = []
  for name in names:
    
    fname = wget.download('https://www.sec.gov/Archives/edgar/data/' + name.split('data_')[1].replace('_','/'), './data')
    
    os.rename(fname, fname + '.orig')
    with open(fname + '.orig', 'rb') as fin, open(fname, 'wb') as fout:
      text = fin.read()
      text = re.sub(r'<IMS-HEADER>.*?</IMS-HEADER>', '', text, flags=re.DOTALL)
      text = re.sub(r'<SEC-HEADER>.*?</SEC-HEADER>', '', text, flags=re.DOTALL)
      text = re.sub(r'<GRAPHIC>.*?</GRAPHIC>', '', text, flags=re.DOTALL)
      text = re.sub(r'<ZIP>.*?</ZIP>', '', text, flags=re.DOTALL)
      text = re.sub(r'<EXCEL>.*?</EXCEL>', '', text, flags=re.DOTALL)
      text = re.sub(r'<PDF>.*?</PDF>', '', text, flags=re.DOTALL)
      text = re.sub(r'<XBRL>.*?</XBRL>', '', text, flags=re.DOTALL)
      fout.write(text)

    text = open(fname).read()
    toks = np.array(nltk.word_tokenize(text))
    
    # Clean words that contain numbers or special characters
    special_inds = [not bool(re.search('[\d\/\*\'\-,=;:@<>\.]', x)) for x in toks]  
    toks = toks[np.array(special_inds)]
    
    # Clean single char words
    word_inds = [len(x)>1 for x in toks]
    toks = toks[np.array(word_inds)]
    
    toks_total.append(sorted(toks))
    print 'N_words:        ' + str(len(toks))
    
    # add negation here
    
    toks = list(set(toks))
    toks_unique.append(sorted(toks))
    print 'N_Unique_Words: ' + str(len(toks))
  
  data['Toks_Total'] = toks_total
  data['Toks_Unique'] = toks_unique

  print data.Toks_Unique
  #print data.columns.values
   

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

  #dictionary = pd.read_excel('data/LoughranMcDonald_MasterDictionary_2014.xlsx', sheetname=0)
  #print dictionary.columns.values
  #print dictionary.head
  
