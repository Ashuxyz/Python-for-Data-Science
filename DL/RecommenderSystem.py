# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 12:55:11 2019

@author: A.Kumar
"""

# Building Recommender system

import pandas as pd
import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens(min_rating=4.0)

# splitting in training / test  datasets
print(repr(data['train']))

print(repr(data['test']))

#Generate model

model = LightFM(loss='warp')

 #Train model

model.fit(data['train'], epochs=30, num_threads=2)

def sample_reco(model, data, user_ids):
    #no of users and movies in training data
    n_users, n_items = data['train'].shape
    for user_id in user_ids:
        #movies they already like
        known_positive = data['item_labels'][data['train'].tocsr()[user_id].indices]
        print(known_positive)
        #movies our model predicted
        scores = model.predict(user_id, np.arange(n_items))
        print(scores)
        #rank movies
        top_items = data['item_labels'][np.argsort(-scores)]
        print(top_items)
        #resulets
        print('\n')
        print("User %s" % user_id)
        print("known positives:")
        for x in known_positive[:3]:
            print('\n %s' % x)

    print("Recomended:")

    for x in top_items[:3]:
        print('\n %s' % x)

sample_reco(model, data, [1,2,3])

