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

def delta_logclose(Close, dt):
  lD = []
  for t in range(dt, len(Close)):
     lD.append(np.log(Close[t] / Close[t-dt]))
  return lD

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
  Close = data.T.values[0]

  # Calculate price differences
  lD  = delta_logclose(Close, 1)
  plt.figure()
  n, bins, patches = plt.hist(lD, normed=True, bins=60, log=False, label=r'$\Delta t = 1 \rm{\ Day}$')
  xt = plt.xticks()[0]
  xmin, xmax = min(xt), max(xt)
  x = np.linspace(xmin, xmax, len(lD))
  m, s = stats.norm.fit(lD)
  pdf_g = stats.norm.pdf(x, m, s)
  plt.plot(x, pdf_g, label='Gaussian fit')
  plt.xlabel(r'$\rm{Log[Close}(t)] - \rm{Log[Close}(t - \Delta t)]\rm{\ (USD)}$', horizontalalignment='right', x=1.0, fontsize=18)
  plt.ylabel('Normalized count', horizontalalignment='right', y=1.0, fontsize=18)
  plt.title(args.ticker + ' ' + args.start_date + ' to ' + args.end_date, fontsize=18)
  plt.legend(loc='best', frameon=False)
  x1, x2, y1, y2 = plt.axis()
  plt.axis([-0.1, 0.1, y1, y2])
  plt.savefig('plots/logfit_'+args.ticker+'.png')
  plt.show()
