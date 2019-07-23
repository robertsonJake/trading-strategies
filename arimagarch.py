# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 22:06:13 2019

@author: Jake
"""

import pandas as pd
import numpy as np
import datetime
import time
import seaborn as sns
import pandas_datareader.data as web
import matplotlib.pyplot as pp
import scipy.stats as stats
import arch
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import math

end = datetime.datetime.today()
start = end - datetime.timedelta(days=365*10)

names = [#'AAPL',
         #'MSFT']
         #'V',
         #'NVDA',
         #'MCD',
         'DIS'
         #'SBUX',
         #'BRK.B',
         #'JNJ',
         #'MA',
         #'PEP',
         #'NFLX',
         #'WM',
         #'GOOG',
         #'AMZN',
         #'BRKB',
         #'WDAY',
         #'CRM',
         #'DATA',
         #'AYX',
         #'SQ',
         #'CHD',
         #'NEE',
         #'VOO',
         #'S&P500'
        ]

df = web.DataReader('AAPL','yahoo',start=start,end=end)

prices = df['Adj Close']
changes = (prices / prices.shift(1)).dropna()
log_returns = np.log(changes)
arch_model = arch.arch_model(log_returns*100,mean='ARX',dist='studentst',vol='GARCH',p=1,q=5)
arch_model = arch_model.fit()
forecasts = arch_model.forecast(horizon=255,method='simulation',
                                start=max(changes.index),simulations=int(1e3))
sims = forecasts.simulations

pp.plot(log_returns,color='#002868')
pp.show()

sns.distplot(log_returns*100,color='#002868')
mu = changes.mean()
std = changes.std()
value = np.random.normal(loc=mu,scale=std,size=1000)
tshape = stats.t.fit(log_returns*100)
t = stats.t.rvs(tshape[0],tshape[1],tshape[2],int(1e6))
#sns.distplot(value)
sns.distplot(t,color='#9cb2d6')
pp.show()

lines = pp.plot(sims.residual_variances[-1,::10].T, color='#9cb2d6')
lines[0].set_label('Simulated path')
line = pp.plot(forecasts.variance.iloc[-1].values, color='#002868')
line[0].set_label('Expected variance')
legend = pp.legend()
pp.show()
sims = forecasts.simulations

lines = pp.plot(sims.values[-1,::10].T, color='#9cb2d6')
lines[0].set_label('Simulated path')
line = pp.plot(forecasts.mean.iloc[-1].values, color='#002868')
line[0].set_label('Expected mean')
legend = pp.legend()
pp.show()


exp = np.vectorize(lambda x: math.exp(x))
pp.plot(prices.iloc[-1]*exp(np.cumsum(sims.values[-1] / 100,axis=1).T),alpha=0.25)
mean_sims = np.mean(prices.iloc[-1]*exp(np.cumsum(sims.values[-1] / 100,axis=1)),axis=0)
std_sims = np.std(prices.iloc[-1]*exp(np.cumsum(sims.values[-1] / 100,axis=1)),axis=0)

pp.plot(mean_sims,color='red')
pp.plot(mean_sims + 2*std_sims,color='black',linestyle='--')
pp.plot(mean_sims - 2*std_sims,color='black',linestyle='--')
pp.show()

def plot_conf(init,m,s,k,start_date):
    mean_returns = np.cumsum([m for i in range(k)])
    pp.plot([start_date + datetime.timedelta(days=i) for i in range(k)],init*exp(np.cumsum([m for i in range(k)])),color='r')
    pp.plot([start_date + datetime.timedelta(days=i) for i in range(k)],init*exp([mean_returns[i]+(2*s*math.sqrt(i)) for i in range(len(mean_returns))])
                                    ,color='black',linestyle='--')
    pp.plot([start_date + datetime.timedelta(days=i) for i in range(k)],init*exp([mean_returns[i]-(2*s*math.sqrt(i)) for i in range(len(mean_returns))])
                                    ,color='black',linestyle='--')

amt = prices.iloc[-1]
n = int(1e5)
print(n)
#sns.distplot(log_returns)
mu = log_returns.mean()
std = log_returns.std()
#value = np.random.normal(loc=mu,scale=std,size=1000)
#tshape = stats.t.fit(log_returns*100)
kdays = 253
exp = np.vectorize(lambda x: math.exp(x))
dingouts = []
for i in range(n):
    log_returns = stats.t.rvs(tshape[0],tshape[1],tshape[2],kdays)
    pf_value = [amt]
    pf_value.extend(amt*exp(np.cumsum(log_returns)))
    #pp.plot(range(kdays+1),pf_value,color='#9cb2d6',alpha=0.025)
    
pp.plot(prices.iloc[-252:])
plot_conf(amt,mu,std,kdays,max(prices.index)+datetime.timedelta(days=1))
pp.show()