# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 06:13:40 2019

@author: A.Kumar
"""

#ARIMA time series analysis for shampoo sale

import pandas as pd
from pandas import datetime
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from pandas.tools.plotting import autocorrelation_plot
import statsmodels.api as sm
from pylab import rcParams
import warnings
import itertools
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
 
datafile='C:/Users/a.kumar/Documents/LEARNING/DataScience/ShampooSale/shampoo-sales.csv'

series = pd.read_csv(datafile)

series.columns=['month','sales']

def parser(x):
    datetime_object = datetime.strptime(x, '%d-%b').date()
    new_date = datetime_object + relativedelta(years=datetime_object.month)
    new_date=pd.to_datetime(new_date,format='%Y-%d-%m')
    return(new_date)
    
for i in series.index:
    changed_date=parser(series['month'][i])
    series['month'][i]=changed_date

series.set_index('month',inplace=True)

series.plot()
plt.xlabel('Year')
plt.show()



autocorrelation_plot(series)
plt.show()

rcParams['figure.figsize'] = 7, 6
decomposition = sm.tsa.seasonal_decompose(series, model='additive')
fig = decomposition.plot()
plt.show()


series.rolling(12).mean().plot(figsize=(10,6), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);

series.diff().plot(figsize=(10,6), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
# get the AIC value and selct the p,d,q for the lowest AIC
for param in pdq:
    
    try:
        mod = ARIMA(series,order=param)
        results = mod.fit()
        print('ARIMA{} - AIC:{}'.format(param,results.aic))
    except:
            continue

# fitting the model
mod_ARIMA = ARIMA(series,order=(1, 1, 1))
results = mod_ARIMA.fit()
print(results.summary().tables[1])

prediction=results.predict()

# prediction the model
X=series.values
size=int(len(X)*0.66)
train,test=X[0:size],X[size:len(X)]

history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(1,1,1))
    model_fit=model.fit(disp=0)
    output=model_fit.forecast()
    yhat=output[0]
    predictions.append(yhat)
    obs=test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()











 


