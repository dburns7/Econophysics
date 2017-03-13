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

  data = pd.read_csv('data/cleaned_10X.csv')
 
  print data.Neg_Counts[data.Neg_Counts > 0]
  print data.Excess_Return[data.Neg_Counts>0] 
  plt.figure()
  ax = sns.regplot(x="Neg_Yield", y="Excess_Return", data=data, ci=68)
  plt.show()
