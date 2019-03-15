# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 12:05:55 2019

@author: A.Kumar
"""

# Bootstrapping for Linear Regression

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

datafile='C:/Users/a.kumar/Documents/LEARNING/Data Science/Datasets/LinearExample.csv'
lin_df=pd.read_csv(datafile)
y=np.array(lin_df['y'])
X=np.array(lin_df.drop('y',axis=1))
X=X.ravel()

def draw_bs_pairs_linreg(X, y, size=1):
    inds = np.arange(len(X))
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)
    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = X[bs_inds], y[bs_inds]
        print(i,':',bs_x, bs_y)
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)

    return bs_slope_reps, bs_intercept_reps

# Generate replicates of slope and intercept using pairs bootstrap
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(X, y, size=4)
print(bs_slope_reps,':',bs_intercept_reps)

