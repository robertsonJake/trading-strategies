# -*- coding: utf-8 -*-
"""
Created on Fri May 31 18:38:57 2019

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

end = pd.to_datetime('01-01-2014')
start = end - datetime.timedelta(days=365*4)

#price = df['Adj Close']

niter = 1e5
mean = 0.05 / 253
stdev = 0.23 / math.sqrt(253)

stopout_dings = []
takeprofit_dings = []
stopout_price = np.log(950000)
takeprofit_price = np.log(1100000)

for i in range(int(niter)):
    #Generate 45 sampels of returns from a 'normal' distribution of log returns
    log_returns = np.random.normal(loc=mean,scale=stdev,size=100)
    logprices = np.log(1e6) + np.cumsum(log_returns)
    logprices = logprices.tolist()
    min_price = min(logprices)
    max_price = max(logprices)
    if min_price < stopout_price and max_price < takeprofit_price:
        stopout_dings.append(i) #If our simulation gets stopped out
    elif max_price >= takeprofit_price and min_price > stopout_price:
        takeprofit_dings.append(i)
    elif max_price >= takeprofit_price and min_price < stopout_price:
        if logprices.index(max_price) < logprices.index(min_price):
            takeprofit_dings.append(i)
            
print('profitable',float(len(takeprofit_dings)) / niter)
print('losses',float(len(stopout_dings)) / niter)

n = 253
mean = 0.1 / 253
stdev = 0.2 / math.sqrt(253)
takeprofit_dings = []
mean_returns = np.cumsum([mean for i in range(n)])
pp.plot(range(n),100*exp(np.cumsum([mean for i in range(n)])),color='r')
pp.plot(range(n),100*exp([mean_returns[i]+(stdev*math.sqrt(i)) for i in range(len(mean_returns))])
                                ,color='black',linestyle='--')
pp.plot(range(n),100*exp([mean_returns[i]-(stdev*math.sqrt(i)) for i in range(len(mean_returns))])
                                ,color='black',linestyle='--')
for i in range(int(20)):
    log_returns = np.random.normal(loc=mean,scale=stdev,size=n)
    exp = np.vectorize(lambda x: math.exp(x))
    prices = 100*exp(np.cumsum(log_returns))
    final_price = prices[-1]
    if final_price > 110:
        takeprofit_dings.append(i)
    pp.plot(range(len(prices)),prices,alpha=0.4)
pp.show()
#print('probability over $110: ',float(len(takeprofit_dings)) / niter)



