# -*- coding: utf-8 -*-
"""
Created on Sun May 26 13:21:23 2019

@author: Jake
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as pp
import datetime
import time
import pandas_datareader.data as web
import seaborn as sns

end = datetime.datetime.today()
start = end - datetime.timedelta(days=365*2)
#df = web.DataReader(['WM','NEE','MSFT','SPY'],'yahoo',start=start,end=end)
#df['Date'] = [mdates.date2num(d) for d in df.index]
df = web.DataReader('WM','yahoo',start=start,end=end)

price = df['Adj Close']
twelve_day = price.ewm(span=12).mean()
twentysix_day = price.ewm(span=26).mean()
macd = twelve_day - twentysix_day
signal = macd.ewm(span=9).mean()

pp.figure(figsize=(15,10))
pp.plot(price,color='green',linewidth=2.0)
pp.grid()
pp.show()

pp.figure(figsize=(15,5))
pp.plot(macd,label='macd',)
pp.plot(signal,label='signal')
pp.axhline(y=0,color='black')
pp.grid()
pp.legend()
pp.show()

crossovers = macd - signal
pp.hist(crossovers,density=True)
sns.kdeplot(crossovers.values)
pp.show()

pp.plot(crossovers)
pp.show()