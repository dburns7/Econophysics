#!/bin/python
# -----------------------------------------------------------------------------
#  File:        TrailingReturns.py
#  Usage:       python TrailingReturns.py 
#  Description: Project to analyze the trailing returns for different stocks
#               over various time scales and compute the largest drawdown.
#  Created:     31-Dec-2016 Dustin Burns
# -----------------------------------------------------------------------------
import pandas as pd
from yahoo_finance import Share
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
import matplotlib.mlab as mlab

def delta_close(Close, dt):
  D = []
  for t in range(dt, len(Close)):
     D.append(Close[t] - Close[t-dt])
  return D

def delta_logclose(Close, dt):
  lD = []
  for t in range(dt, len(Close)):
     lD.append(np.log(Close[t]) - np.log(Close[t-dt]))
  return lD

def trailing_return(Close, dt):
  R = []
  for t in range(dt, len(Close)):
     R.append(100 * (Close[t] - Close[t-dt])/Close[t-dt])
  return [float('NaN')]*dt + R

def gauss(x, mu, sigma):
  return((np.exp(-(x-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))))

if __name__ == "__main__":

  # Parse command line arguments
  parser = argparse.ArgumentParser(description="Compute and plot trailing returns")
  parser.add_argument('--ticker', nargs='?', default='YHOO', help='Fund ticker symbol')
  parser.add_argument('--start_date', nargs='?', default='2013-04-25')
  parser.add_argument('--end_date', nargs='?', default='2017-01-10')
  args = parser.parse_args()

  # Get data
  share = Share(args.ticker)
  data  = share.get_historical(args.start_date, args.end_date)
  data  = pd.DataFrame(data)
  data.Date  = pd.to_datetime(data.Date)
  data.set_index(data.Date, inplace=True)
  data.sort_index(inplace=True)
  data.Adj_Close = data.Adj_Close.astype(float)

  print data.columns.values
 
  Close = data.T.values[0]
  print 'Mean: ' + str(np.mean(Close))
  print 'Sigma: ' + str(np.std(Close))
 
  # Calculate trailing returns
  R_3mo  = trailing_return(Close, 21*3)
  R_12mo = trailing_return(Close, 21*12)
  data['R_3mo']  = R_3mo
  data['R_12mo'] = R_12mo 
  print '3 month maximum drawdown: ' + str(np.nanmin(R_3mo))
  print '12 month maximum drawdown: ' + str(np.nanmin(R_12mo))

  plt.figure()
  data.plot(y='R_3mo', label=r'$\Delta t = 3 \rm{\ Month}$')
  ax = data.plot(y='R_12mo', label=r'$\Delta t = 12 \rm{\ Month}$')
  ax.set_ylabel('Trailing Return %', horizontalalignment='right', y=1.0, fontsize=18)
  ax.set_xlabel('')
  plt.title(args.ticker, fontsize=18)
  plt.legend(loc='best', frameon=False)
  plt.savefig('plots/TrailingReturns_'+args.ticker+'.png')

  plt.show()

  '''
  # Calculate price differences
  D_3mo  = delta_close(Close, 21*3)
  D_12mo = delta_close(Close, 21*12)
  plt.figure()
  n, bins, patches = plt.hist(D_3mo, normed=True, bins=60, label=r'$\Delta t = 3 \rm{\ Month}$')
  #plt.hist(D_12mo, normed=True, bins=30, label=r'$\Delta t = 12 \rm{\ Month}$')
  #mu = np.nanmean(D_3mo)
  #sigma = np.nanvar(D_3mo)
  #x = np.linspace(min(D_3mo), max(D_3mo), 100)
  #plt.plot(x, mlab.normpdf(x, mu, np.sqrt(sigma)), label='Gaussian fit')
  xt = plt.xticks()[0]
  xmin, xmax = min(xt), max(xt)
  x = np.linspace(xmin, xmax, len(D_3mo))
  m, s = stats.norm.fit(D_3mo)
  pdf_g = stats.norm.pdf(x, m, s)
  plt.plot(x, pdf_g, label='Gaussian fit')
  plt.xlabel(r'$\rm{Close}(t) - \rm{Close}(\Delta t)\rm{\ [USD]}$', horizontalalignment='right', x=1.0, fontsize=18)
  plt.ylabel('Normalized count', horizontalalignment='right', y=1.0, fontsize=18)
  plt.legend(loc='best', frameon=False)
   
 
  # Calculate differences of logs of prices
  D_3mo  = delta_logclose(Close, 21*3)
  D_12mo = delta_logclose(Close, 21*12)
  plt.figure()
  n, bins, patches = plt.hist(D_3mo, normed=True, bins=60, label=r'$\Delta t = 3 \rm{\ Month}$')
  xt = plt.xticks()[0]
  xmin, xmax = min(xt), max(xt)
  x = np.linspace(xmin, xmax, len(D_3mo))
  a,b,c = stats.lognorm.fit(D_3mo)
  pdf_g = stats.lognorm.pdf(x, a, b, c)
  ax = plt.plot(x, pdf_g, label='Lognormal fit')
  #ax.set_autoscale_on(False)
  plt.xlabel(r'$\log\rm{Close}(t) - \log\rm{Close}(\Delta t)\rm{\ [USD]}$', horizontalalignment='right', x=1.0, fontsize=18)
  plt.ylabel('Normalized count', horizontalalignment='right', y=1.0, fontsize=18)
  plt.legend(loc='best', frameon=False)
  plt.axis([-1, 1, 0, 10])




  plt.figure()
  ax = data.plot(y='Close')
  ax.set_ylabel('Daily Closing Price [USD]', horizontalalignment='right', y=1.0, fontsize=18)
  ax.set_xlabel('')
  plt.title(args.ticker, fontsize=18)
  plt.savefig('plots/Close_'+args.ticker+'.png')
  
  
  plt.figure()
  ax = data.R_3mo.hist(bins=60, label=r'$\Delta t = 3 \rm{\ Month}$')
  ax = data.R_12mo.hist(bins=60, label=r'$\Delta t = 12 \rm{\ Month}$')
  plt.title(args.ticker, fontsize=18)
  ax.set_xlabel('Trailing Return %', horizontalalignment='right', x=1.0, fontsize=18)
  ax.set_ylabel('Count', horizontalalignment='right', y=1.0, fontsize=18)
  plt.legend(loc='best', frameon=False)
  plt.savefig('plots/R_'+args.ticker+'.png')


  Drawback_iter = []
  Drawback_err  = []
  for i in range(0, len(Close)/21):
    tr = np.array(trailing_return(Close, 21*i))[~np.isnan(trailing_return(Close, 21*i))]
    Drawback_iter.append(np.nanmean(tr))
    Drawback_err.append(stats.sem(tr))
    #Drawback_iter.append(np.nanmin(trailing_return(Close, i)))
  plt.figure()
  #plt.yscale('log')
  plt.title(args.ticker, fontsize=18)
  ax = plt.errorbar(range(0, len(Close)/21), Drawback_iter, yerr=Drawback_err)
  plt.xlabel(r'$\Delta t \rm{\ [Month]}$', horizontalalignment='right', x=1.0, fontsize=18)
  plt.ylabel(r'$\rm{mean}[R(t, \Delta t)]$', horizontalalignment='right', y=1.0, fontsize=18)
  plt.savefig('plots/meanR_'+args.ticker+'.png')


  # Format plots
  plt.figure(1)
  plt.subplot(211)
  plt.title(args.ticker)
  plt.plot(Date[21*3-1:-1], R_3mo, 'b')
  plt.plot(Date[21*3-1:-1], [min(R_3mo)]*len(Date[21*3-1:-1]), 'r--', label='Largest Drawdown: ' + '%.2f' % min(R_3mo))
  #plt.axis([min(Date), max(Date), min([-1, min(R_3mo)]), max([1, max(R_3mo)])])
  plt.ylabel('3 Month Trailing Returns')
  plt.grid(True)
  plt.legend(loc='lower right', frameon=False, prop={'size':12})
  plt.subplot(212)
  plt.plot(Date[21*12-1:-1], R_12mo, 'b')
  #plt.plot(Date[21*12-1:-1], [min(R_12mo)]*len(Date[21*12-1:-1]), 'r--', label='Largest Drawdown: ' + '%.2f' % min(R_12mo))
  #plt.axis([min(Date), max(Date), min([-1, min(R_12mo)]), max([1, max(R_12mo)])])
  plt.ylabel('12 Month Trailing Returns')
  plt.xlabel('Unix Time [days]')
  plt.grid(True)
  plt.legend(loc='lower right', frameon=False, prop={'size':12})
  plt.savefig('plots/TrailingReturns_'+args.ticker+'.png')

  plt.figure(2)
  plt.yscale('log')
  n, bins, patches = plt.hist(R_3mo, 50)
  plt.xlabel(r'$R(t, \Delta t = \mathrm{3 months})$')
  plt.savefig('plots/hist_TrailingReturns_'+args.ticker+'.png')
  
  #plt.figure(3)
  #plt.plot(Drawback_iter)
  #plt.axis([0, len(Close), -1, 1]) 
  '''
  
