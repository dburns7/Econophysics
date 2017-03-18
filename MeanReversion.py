#!/bin/python
# -----------------------------------------------------------------------------
#  File:        MeanReversion.py
#  Usage:       python MeanReversion.py
#  Description: Project to optimize parameters of mean reversion investment strategy
#  Created:     15-2-2017 Dustin Burns
# -----------------------------------------------------------------------------
import pandas as pd
from yahoo_finance import Share
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.fftpack
from scipy.optimize import curve_fit
import matplotlib.mlab as mlab
import savitzky_golay

def delta_close(Close, dt):
  D = []
  for t in range(dt, len(Close)):
     D.append(Close[t] - Close[t-dt])
  return D

def TrailingAvg(Close, dt):
  Trailing_Avg = []
  for i in range(dt, len(Close)):
    Trailing_Avg.append(sum(Close[i-dt:i]) / dt)
  return [float('NaN')]*dt + Trailing_Avg

def TrailingStd(Close, dt):
  Trailing_Std = []
  for i in range(dt, len(Close)):
    Trailing_Std.append(np.std(Close[i-dt:i]))
  return [float('NaN')]*dt + [x / max(Trailing_Std) for x in Trailing_Std]

def CrossingPoints(Date, Delta_Close_Avg, Close):
  CrossingPoints = []
  y = []
  yClose = []
  for i in range(0, len(Delta_Close_Avg)):
    if abs(Delta_Close_Avg[i]) < 0.1: 
      CrossingPoints.append(Date[i])
      y.append(Delta_Close_Avg[i])
      yClose.append(Close[i])
  return (y, yClose, CrossingPoints)

def smooth(y):
  return savitzky_golay.savitzky_golay(y, 101, 4)

#find local min and max
def extrema(x, y):  
  min_ind = (np.diff(np.sign(np.diff(y))) > 0).nonzero()[0] + 1
  max_ind = (np.diff(np.sign(np.diff(y))) < 0).nonzero()[0] + 1
  xmin = []
  xmax = []
  ymin = []
  ymax = []
  for i in range(0, len(min_ind)): 
    xmin.append(x[min_ind[i]])
    ymin.append(y[min_ind[i]])
  for i in range(0, len(max_ind)):
    xmax.append(x[max_ind[i]])
    ymax.append(y[max_ind[i]])
  return (xmin, ymin, xmax, ymax, min_ind, max_ind)


if __name__ == "__main__":

  # Parse command line arguments
  parser = argparse.ArgumentParser(description="Compute and plot trailing returns")
  parser.add_argument('--ticker', nargs='?', default='YHOO', help='Fund ticker symbol')
  parser.add_argument('--start_date', nargs='?', default='2013-12-01')
  parser.add_argument('--end_date', nargs='?', default='2017-02-20')
  args = parser.parse_args()

  # Get data
  share = Share(args.ticker)
  data  = share.get_historical(args.start_date, args.end_date)
  data  = pd.DataFrame(data)
  data.Date  = pd.to_datetime(data.Date)
  data.set_index(data.Date, inplace=True)
  data.sort_index(inplace=True)
  data.Adj_Close = data.Adj_Close.astype(float)
  #print data.columns.values

 
  #Close = data.Adj_Close
  dt = 200
  data['Trailing_Avg'] = TrailingAvg(data.Adj_Close, dt)
  data['Trailing_Std'] = TrailingStd(data.Adj_Close, dt)
 
  # smooth curve with low pass filter
  data['Close_smooth'] = smooth(data.Adj_Close)

  # calculate and regularize difference in closing price and trailing avg 
  data['Delta_Close_Avg'] = data['Close_smooth'] - data['Trailing_Avg']
  #data.Delta_Close_Avg = np.abs(data.Delta_Close_Avg - data.Delta_Close_Avg.mean()) / data.Delta_Close_Avg.std()
  #data['Delta_Close_Avg'] = np.abs(data['Adj_Close'] / data['Trailing_Avg'] - 1)
 
  # get local min and max off smoothed delta
  data['Delta_smooth'] = smooth(data.Delta_Close_Avg)
  xmin, ymin, xmax, ymax, min_ind, max_ind = extrema(data.Date, data.Delta_Close_Avg)
  #xmin, ymin, xmax, ymax, min_ind, max_ind = extrema(data.Date, data.Delta_smooth)
  cmin = []
  cmax = []
  for i in range(0, len(min_ind)): cmin.append(data.Adj_Close[min_ind[i]])
  for i in range(0, len(max_ind)): cmax.append(data.Adj_Close[max_ind[i]])


  y, yClose, CrossingPoints = CrossingPoints(data.Date, data.Delta_Close_Avg, data.Adj_Close)
  
  # plot close-avg, smoothed, and extrema
  plt.figure()
  ax = data.plot(y='Delta_Close_Avg', logy=False, style='b', label='|Close - Trailing_Avg|')
  #ax = data.plot(y='Delta_smooth', logy=True, style='r', label='Smoothed')
  plt.plot(CrossingPoints, y, 'ro')
  #plt.scatter(xmin, ymin, s=60, c='r', marker='v', label='Local minima')
  #plt.scatter(xmax, ymax, s=60, c='r', marker='^', label='Local maxima')
  ax.set_ylabel('Price (USD)', horizontalalignment='right', y=1.0, fontsize=18)
  ax.set_xlabel('')
  plt.title(args.ticker, fontsize=18)
  plt.legend(loc='best', frameon=False)
  
  '''
  # Plot trailing std with local min and max 
  plt.figure()
  plt.scatter(x1, d1, s=60, c='r', marker='v', label='Local minima')
  plt.scatter(x2, d2, s=60, c='r', marker='^', label='Local maxima')
  ax = data.plot(y='Trailing_Std', logy=False, label='Std')
  data.plot(y='smooth_std', label='smooth std')
  ax.set_ylabel('Price (USD)', horizontalalignment='right', y=1.0, fontsize=18)
  ax.set_xlabel('')
  plt.title(args.ticker, fontsize=18)
  plt.legend(loc='best', frameon=False)
  '''
  
  # plot close, avg, and critical points
  plt.figure()
  ax = data.plot(y='Adj_Close', label='Close')
  data.plot(y='Trailing_Avg', label='200 day trailing average')
  plt.plot(CrossingPoints, yClose, 'ro')
  #plt.scatter(xmin, cmin, s=60, c='r', marker='v', label='Local minima')
  #plt.scatter(xmax, cmax, s=60, c='r', marker='^', label='Local maxima')
  ax.set_ylabel('Price (USD)', horizontalalignment='right', y=1.0, fontsize=18)
  ax.set_xlabel('')
  plt.title(args.ticker, fontsize=18)
  plt.legend(loc='best', frameon=False)
  plt.savefig('plots/Close_'+args.ticker+'.png')

  plt.show()
