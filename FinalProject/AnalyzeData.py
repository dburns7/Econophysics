#!/bin/python
# -----------------------------------------------------------------------------
#  File:        AnalyzeData.py
#  Usage:       python AnalyzeData.py 
#  Description: Analyze cleaned 10-X data with linear regression and visualizations
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

if __name__ == "__main__":

  data1 = pd.read_csv('data/cleaned_10X_1.csv')
  data2 = pd.read_csv('data/cleaned_10X_2.csv')
  data3 = pd.read_csv('data/cleaned_10X_3.csv')
  data4 = pd.read_csv('data/cleaned_10X_4.csv')
  data5 = pd.read_csv('data/cleaned_10X_5.csv')
  #data6 = pd.read_csv('data/cleaned_10X_6.csv')
  data7 = pd.read_csv('data/cleaned_10X_7.csv')
  data8 = pd.read_csv('data/cleaned_10X_8.csv')
  data9 = pd.read_csv('data/cleaned_10X_9.csv')
  data10 = pd.read_csv('data/cleaned_10X_10.csv')
  data11 = pd.read_csv('data/cleaned_10X_11.csv')
  data12 = pd.read_csv('data/cleaned_10X_12.csv')
  data13 = pd.read_csv('data/cleaned_10X_13.csv')
  data14 = pd.read_csv('data/cleaned_10X_14.csv')
  data15 = pd.read_csv('data/cleaned_10X_15.csv')
  data = pd.concat([data1, data2, data3, data4, data5, data7, data8, data9, data10, data11, data12, data13, data14, data15])
  #data = pd.read_csv('data/cleaned_10X_100.csv')
  data['Prop_Yield_Paper'] = data.N_Negative_Paper / data.N_Words_Paper
 
  (a, b, r, tt, stderr) = stats.linregress(data.Neg_Yield, data.Excess_Return)
  print 'Slope:           ' + str(a)
  print 'Standard error:  ' + str(stderr)
  print 'R-squared:       ' + str(r**2)
  print 'p-value:         ' + str(tt)
  
  plt.figure()
  #ax = sns.regplot(x="Prop_Yield_Paper", y="Excess_Return", data=data, x_bins=5, ci=68)
  ax = sns.regplot(x="N_Negative_Paper", y="Excess_Return", data=data, x_bins=5, ci=68, x_estimator=np.mean)
  #ax = sns.regplot(x="Neg_Total", y="Excess_Return", data=data, ci=68, x_estimator=np.median)
  #ax = sns.regplot(x="Neg_Yield", y="Excess_Return", data=data, ci=68, x_bins=5, x_estimator=np.mean)
  #ax = sns.regplot(x="Neg_Yield", y="Excess_Return", data=data, x_bins=5, ci=68, x_estimator=np.mean)
  #plt.xlabel('Weighted negative word count', horizontalalignment='right', x=1.0, fontsize=16)
  plt.xlabel('Unweighted negative word count', horizontalalignment='right', x=1.0, fontsize=16)
  #plt.xlabel('tf.idf weighted negative word count', horizontalalignment='right', x=1.0, fontsize=16)
  #plt.xlabel('Proportional weighted negative word count', horizontalalignment='right', x=1.0, fontsize=16)
  #plt.ylabel('Filing period excess return (%)', horizontalalignment='right', y=1.0, fontsize=16)
  plt.ylabel('Mean filing period excess return (%)', horizontalalignment='right', y=1.0, fontsize=16)
  #plt.legend(loc='best', frameon=False)
  #x1, x2, y1, y2 = plt.axis()
  #plt.axis([-0.1, 0.1, y1, y2])
  plt.show()
