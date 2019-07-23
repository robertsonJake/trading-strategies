# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 11:34:19 2019

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
start = end - datetime.timedelta(days=365*30)

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

df = web.DataReader('^GSPC','yahoo',start=start,end=end)