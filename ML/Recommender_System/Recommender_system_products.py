# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:23:38 2019

@author: A.Kumar
"""

# Recommendation sytem for products in an online store

import pandas as pd
import numpy as np
import time
import turicreate as tc
from sklearn.model_selection import train_test_split

datafile_customers='C:/Users/a.kumar/Documents/LEARNING/DataScience_and_Analytics/Python-for-Data-Science/ML/Recommender_System/recommend_1.csv'
datafile_transactions='C:/Users/a.kumar/Documents/LEARNING/DataScience_and_Analytics/Python-for-Data-Science/ML/Recommender_System/trx_data.csv'

customers=pd.read_csv(datafile_customers)
transactions=pd.read_csv(datafile_transactions)

# data preprocessing on transactions dataframe

data = pd.melt(transactions.set_index('customerId')['products'].apply(pd.Series).reset_index(), 
             id_vars=['customerId'],
             value_name='products') \
    .dropna().drop(['variable'], axis=1) \
    .groupby(['customerId', 'products']) \
    .agg({'products': 'count'}) \
    .rename(columns={'products': 'purchase_count'}) \
    .reset_index() \
    .rename(columns={'products': 'productId'})

data['productId'] = data['productId'].astype('int64')
