# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 23:21:09 2019

@author: Jake
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
#import time
import pandas_datareader.data as web
import email_info

np.set_printoptions(threshold=np.nan)

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns
  
def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in xrange(num_portfolios):
        weights = np.random.random(len(names))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record

def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    #print "-"*80
    #print "Maximum Sharpe Ratio Portfolio Allocation\n"
    #print "Annualised Return:", round(rp,2)
    #print "Annualised Volatility:", round(sdp,2)
    #print "\n"
    #print max_sharpe_allocation
    #print "-"*80
    #print "Minimum Volatility Portfolio Allocation\n"
    #print "Annualised Return:", round(rp_min,2)
    #print "Annualised Volatility:", round(sdp_min,2)
    #print "\n"
    #print min_vol_allocation
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)
    
import scipy.optimize as sco
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return result

def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients

def display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, _ = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    
    message = "---------------" + " \n Maximum Sharpe Ratio Portfolio Allocation"+" \n Annualised Return:," + str(round(rp,2)) +" \n Annualised Volatility:," + str(round(sdp,2)) +" \n " + max_sharpe_allocation.to_string()
    
    
    #print "-"*80
    #print "Maximum Sharpe Ratio Portfolio Allocation\n"
    #print "Annualised Return:", round(rp,2)
    #print "Annualised Volatility:", round(sdp,2)
    #print "\n"
    #print max_sharpe_allocation
    #print "-"*80
    #print "Minimum Volatility Portfolio Allocation\n"
    #print "Annualised Return:", round(rp_min,2)
    #print "Annualised Volatility:", round(sdp_min,2)
    #print "\n"
    #print min_vol_allocation
    #print(message)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')

    target = np.linspace(rp_min, 0.32, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    plt.title('Calculated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)
    #plt.show()
    plt.savefig('current_max_sharpe_calc.png')
    return message
    

#---------------------------------------------------

end = datetime.datetime.today()
start = end - datetime.timedelta(days=365)

names = [#'AAPL',
         'MSFT',
         'V',
         #'NVDA',
         'MCD',
         'DIS',
         #'SBUX',
         #'BRK.B',
         #'JNJ',
         'MA',
         #'PEP',
         #'NFLX',
         #'WM',
         'GOOG',
         #'AMZN',
         #'BRKB',
         #'WDAY',
         #'CRM',
         #'DATA',
         #'AYX',
         #'SQ',
         #'CHD',
         'NEE',
         'VOO']
         #'S&P500']
dfs = []
for name in names:
    print(name)
    df = web.DataReader(name,'yahoo',start=start,end=end)
    #df.set_index(df.index.levels[1],inplace=True)
    df.sort_index(ascending=True,inplace=True)
    df['Price'] = df['Adj Close']
    df['Price'] = df['Price'].apply(lambda x: float(str(x).replace(',','')))
    df.columns = [name+' '+c for c in df.columns]
    #df = df[name+' Price']
    dfs.append(df)
    #time.sleep(2)
#df = pd.read_csv('AAPL.csv')
    
data = dfs[0]
for i in range(1,len(dfs)):
    data = data.join(dfs[i])
table = data[[name+' Price' for name in names]].dropna().iloc[-252:]

returns = table.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 25000
risk_free_rate = 0.035

output = display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)


import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
#Next, log in to the server
sender_email = email_info.EMAIL
server.login(sender_email, email_info.KEY)

#Send the mail
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEImage import MIMEImage
msg = MIMEMultipart()
msg['From'] = sender_email
msg['To'] = sender_email
msg['Subject'] = "Portfolio Recommendation"
msg.attach(MIMEText(output, 'plain'))
msg.attach(MIMEImage(open('C:\\Users\\Jake\\Documents\\Finance\\current_max_sharpe_calc.png','rb').read()))
server.sendmail(sender_email,sender_email,msg.as_string())