# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 17:16:22 2019

@author: A.Kumar
"""

# Stock predition using LinearRegression

import pandas as pd
import matplotlib.pyplot as plt
#from statsmodel.graphic.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
import itertools
import warnings
warnings.filterwarnings('ignore')
#from sklearn.metrics import mean_square_error


datafile='C:/Users/a.kumar/Documents/LEARNING/DataScience/Stock/Stock.csv'
df=pd.read_csv(datafile)

df_AAPL=df[df.symbol=='AAPL']
df_AAPL.date=pd.to_datetime(df_AAPL.date,format='%Y-%m-%d')
df_AAPL.index=df_AAPL['date']
df_AAPL=df_AAPL.sort_index(ascending=True,axis=0)


df_AAPL_new=df_AAPL[['close']]
df_AAPL_new_resample=df_AAPL_new.resample('M').mean()

plt.figure(figsize=(6,4))
plt.plot(df_AAPL_new_resample['close'], label='Close Price history')

close_diff=df_AAPL_new_resample['close'].diff(periods=1)
close_diff=close_diff.dropna(axis=0)

plt.plot(close_diff)

X=close_diff.values
train=X[0:65]
test=X[65:]

# for AR
model_AR=AR(train)
model_AR_fit=model_AR.fit()
prediction_AR=model_AR_fit.predict(start=65,end=83)

plt.plot(test)
plt.plot(prediction_AR,color='red')


# for ARIMA
p=d=q=range(0,5)
pdq=list(itertools.product(p,d,q))


for param in pdq:
    try:
        model_ARIMA=ARIMA(train,order=param)
        model_ARIMA_fit= model_ARIMA.fit()
        print(param,model_ARIMA_fit.aic)
    except:
        continue
    
optimized_model_ARIMA=ARIMA(train,order=(0,1,2))
optimized_model_ARIMA_fit=optimized_model_ARIMA.fit()
prediction_ARIMA=optimized_model_ARIMA_fit.predict(start=65,end=83)

plt.plot(test)
plt.plot(prediction_ARIMA,color='red')
    

























