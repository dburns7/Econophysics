#!/bin/python
# -----------------------------------------------------------------------------
#  File:        VolatilityCorrelations.py
#  Usage:       python VolatilityCorrelations.py 
#  Description: Project to calculate historical volatility and correlate to returns
#  Created:     7-Mar-2016 Dustin Burns
# -----------------------------------------------------------------------------
import pandas as pd
from yahoo_finance import Share
import argparse
import matplotlib.pyplot as plt
import numpy as np

def volatility(Close):
  _V = []
  for t in range(21, len(Close)):
    v = 0
    for j in range(0, 21):
      v += np.log(Close[t-j] / Close[t-j-1])**2
    _V.append(np.sqrt(252 * v / 21))
  return [float('NaN')]*21 + _V

def trailing_return(Close, dt):
  R = []
  for t in range(dt, len(Close)):
     R.append(1 * (Close[t] - Close[t-dt])/Close[t-dt])
  return [float('NaN')]*dt + R

def mean(x):
  mu = 0
  n = 0
  for i in range(0, len(x)):
    if not np.isnan(x[i]): 
      mu += x[i]
      n += 1
  return mu / n

def std(x, mu):
  sig = 0
  n = 0
  for i in range(0, len(x)):
    if not np.isnan(x[i]):
      sig += (x[i] - mu)**2
      n += 1
  return np.sqrt(sig / n)

def corr(x, y):
  mu_x = mean(x)
  mu_y = mean(y)
  std_x = std(x, mu_x)
  std_y = std(y, mu_y)
  corr = 0
  n = 0
  if not len(x) == len(y): print 'x and y must have same length'
  for i in range(0, len(x)):
    if not np.isnan(x[i]) and not np.isnan(y[i]):
      corr += (x[i] - mu_x) * (y[i] - mu_y)
      n += 1
  return corr / (n * std_x * std_y)
  

if __name__ == "__main__":

  # Parse command line arguments
  parser = argparse.ArgumentParser(description="Compute and plot trailing returns")
  parser.add_argument('--ticker', nargs='?', default='YHOO', help='Fund ticker symbol')
  parser.add_argument('--start_date', nargs='?', default='2013-04-25')
  parser.add_argument('--end_date', nargs='?', default='2017-01-10')
  args = parser.parse_args()

  # Pull data from yahoo_finance, format data
  share = Share(args.ticker)
  data  = share.get_historical(args.start_date, args.end_date)
  data  = pd.DataFrame(data)
  data.Date  = pd.to_datetime(data.Date)
  data.set_index(data.Date, inplace=True)
  data.sort_index(inplace=True)
  data.Adj_Close = data.Adj_Close.astype(float)
  Close = data.Adj_Close

  # Calculate volitility and trailing return
  data['Volatility'] = volatility(Close)
  data['Trailing_Return'] = trailing_return(Close, 21)
 
  # Calculate correlations
  corr = corr(data.Volatility, data.Trailing_Return)
  print 'Correlation coefficient: ' + str(corr)
  
  # Cross check with built-in function
  corr_check = np.corrcoef(data.Volatility[21:-1], data.Trailing_Return[21:-1])
  print 'Cross check: ' + str(corr_check[0][1])
 
  # Plot volatility and trailing return
  plt.figure()
  ax = data.plot(y='Volatility', label=r'Volatility $\Delta t = 21$ Days')
  data.plot(y='Trailing_Return', label=r'Fractional trailing return $\Delta t = 21$ Days')
  ax.set_ylabel('')
  ax.set_xlabel('')
  plt.title(args.ticker, fontsize=18)
  plt.legend(loc='best', frameon=False)
  plt.savefig('plots/Volatility_'+args.ticker+'.png')
  plt.show()
