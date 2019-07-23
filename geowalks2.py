# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:41:24 2019

@author: Jake
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as pp
import datetime
#import time
#import pandas_datareader.data as web
import seaborn as sns
#from scipy import stats
import math

def plot_conf(init,m,s,k):
    mean_returns = np.cumsum([m for i in range(k)])
    pp.plot(range(k),init*exp(np.cumsum([m for i in range(k)])),color='r')
    pp.plot(range(k),init*exp([mean_returns[i]+(2*s*math.sqrt(i)) for i in range(len(mean_returns))])
                                    ,color='black',linestyle='--')
    pp.plot(range(k),init*exp([mean_returns[i]-(2*s*math.sqrt(i)) for i in range(len(mean_returns))])
                                    ,color='black',linestyle='--')

mu = 0.08 / 253
std = 0.15 / math.sqrt(253)

amt = 80
n = int(1e5)
print(n)
kdays = 253*2
exp = np.vectorize(lambda x: math.exp(x))
dingouts = []
for i in range(n):
    log_returns = np.random.normal(loc=mu,scale=std,size=(kdays))
    pf_value = [amt]
    pf_value.extend(amt*exp(np.cumsum(log_returns)))
    #pp.plot(range(kdays+1),pf_value,alpha=0.03)
    if pf_value[-1] < 90:
        dingouts.append(i)
print('dingout %',float(len(dingouts))/n)
plot_conf(amt,mu,std,kdays)
pp.show()
