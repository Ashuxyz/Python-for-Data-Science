# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 13:08:33 2019

@author: A.Kumar
"""

# Bootstrapping for a dataset

import pandas as pd
import numpy as np

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

bootstrap_count=np.random.randint(1,3)
print(bootstrap_count)


def bootstrap_replicate(data):
    return (data.sample(n=len(data),replace=True))

def draw_bs_reps(data,size=1):

    bs_replicates = []
    
    for i in range(size):
        bs_replicates.append(bootstrap_replicate(data))

    return bs_replicates

print(draw_bs_reps(dataset,size=bootstrap_count))