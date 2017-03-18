#!/bin/python
# -----------------------------------------------------------------------------
#  File:        StatisticalDistributions.py
#  Usage:       python StatisticalDistributions.py 
#  Description: Project to generate and fit the common statistical distributions
#               used to model financial systems.
#  Created:     29-Jan-2017 Dustin Burns
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.misc import factorial

# Guassian deviates, method 1
def pG1(N):
  return np.sum(2*np.random.rand(N)-[1]*N) / np.sqrt(N)

# Gaussian deviates, method 2
def pG2(ndev):
  yG = []
  for i in range(0, ndev):
    x = np.random.rand(2)
    Z0 = np.sqrt(-2*np.log(x[0])) * np.cos(2*np.pi * x[1])
    Z1 = np.sqrt(-2*np.log(x[0])) * np.sin(2*np.pi * x[1])
    yG.append(Z0)
    yG.append(Z1)
  return yG

# Poisson deviates
def pP(mean, ndev):
  pP = []
  for i in range(0, ndev):
    L = np.exp(-mean)
    k = 0
    p = 1
    while p > L:
      k = k + 1
      p *= np.random.rand(1)
    pP.append(k - 1)
  return pP  

# Poisson fit function
def poisson(k, m):
  return m**k * np.exp(-m) / factorial(k)

if __name__ == "__main__":
  
  # Set number of random deviates
  ndev = 100000

  # Gaussian method 1 based on central limit theorem
  y = []
  N = 100 #Number of URNs for each deviate
  for i in range(0, ndev): y.append(pG1(N))
  
  # Gaussian method 2 based on Box-Muller transformation
  yG = pG2(ndev)

  # Format plot
  plt.figure()
  plt.hist(yG, bins=60)
  plt.xlabel('Gaussian random deviates', horizontalalignment='right', x=1.0, fontsize=18)
  plt.ylabel('Count', horizontalalignment='right', y=1.0, fontsize=18)
  plt.savefig('plots/gauss.png')
 
  # Normalize plot and add best fit line 
  plt.figure()
  n, bins, patches = plt.hist(yG, normed=True, bins=60)
  x = np.linspace(-10, 10, 100)
  m, s = stats.norm.fit(yG)
  print 'Gaussian best fit: mean = ' + str(m) + ', sigma = ' + str(s)
  pdf_g = stats.norm.pdf(x, m, s)
  plt.plot(x, pdf_g, linewidth=2, color='r', label='Gaussian best fit')
  plt.xlabel('Gaussian random deviates', horizontalalignment='right', x=1.0, fontsize=18)
  plt.ylabel('Normalized count', horizontalalignment='right', y=1.0, fontsize=18)
  plt.legend(loc='upper right', frameon=False)
  plt.axis([-5, 5, 0, 0.5])
  plt.savefig('plots/gauss_fit.png')
  
  # Generate Poisson random deviates using simple wiki method
  yP = pP(2, 100000)

  # Format plot
  plt.figure()
  plt.hist(yP, bins=20, range=[-0.5, 19.5])
  plt.xlabel('Poisson random deviates', horizontalalignment='right', x=1.0, fontsize=18)
  plt.ylabel('Count', horizontalalignment='right', y=1.0, fontsize=18)
  plt.savefig('plots/poiss.png')

  # Normalize plot and add best fit line
  plt.figure()
  n, bins, patches = plt.hist(yP, normed=True, bins=20, range=[-0.5, 19.5])
  bins = 0.5*(bins[1:] + bins[:-1])
  params, cov = curve_fit(poisson, bins, n)
  print 'Poisson best fit: mean = ' + str(params[0])
  x = np.linspace(0, 20, 100)
  plt.plot(x, poisson(x, *params), linewidth=2, color='r', label='Poisson best fit')
  plt.xlabel('Poisson random deviates', horizontalalignment='right', x=1.0, fontsize=18)
  plt.ylabel('Normalized count', horizontalalignment='right', y=1.0, fontsize=18)
  plt.legend(loc='best', frameon=False)
  plt.axis([-1, 10, 0, 0.3])
  plt.savefig('plots/poiss_fit.png')
  
  plt.show()
