# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:41:27 2019

@author: A.Kumar
"""

# Time series analysis for keyword search 'diet and gym' in google

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pylab import rcParams

datafile='C:/Users/a.kumar/Documents/LEARNING/DataScience/googleTrend/googleTrend.csv'
df=pd.read_csv(datafile)

df.columns=['month','diet','gym','finance']


df['month'] = pd.to_datetime(df['month'])

print(df['month'].min())
print(df['month'].max())

df.set_index('month',inplace=True)

y_diet=df[['diet']]
y_gym=df[['gym']]
y_finance=df[['finance']]

# plot for all 3 columns together
df.plot(figsize=(10,6), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);


# plot for each column separte
y_diet.plot(figsize=(10,6), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);

#showing trend,seasonal and noise all together for single column
rcParams['figure.figsize'] = 7, 6
decomposition = sm.tsa.seasonal_decompose(y_diet, model='additive')
fig = decomposition.plot()
plt.show()

# identifying trends/seasonality in time series using 
#rolling average to smooth the noise and trend
y_diet.rolling(12).mean().plot(figsize=(10,6), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);

y_gym.rolling(12).mean().plot(figsize=(10,6), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);

# plotting rolling mean for both diet and gym together
df_rm = pd.concat([y_diet.rolling(12).mean(), y_gym.rolling(12).mean()], axis=1)
df_rm.plot(figsize=(10,6), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);

# differencing is a way to remove the trend to make it stationary
y_diet.diff().plot(figsize=(10,6), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);


# coefficient for df
df.corr()
df.diff().plot(figsize=(10,6), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);

#coefficients after differencing
df.diff().corr()

pd.plotting.autocorrelation_plot(y_diet);




