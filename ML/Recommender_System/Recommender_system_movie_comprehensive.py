# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 11:48:07 2019

@author: A.Kumar
"""

# Recommender system in a comprehensive way for movie data and user demography

import pandas as pd
import numpy as np

#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('C:/Users/a.kumar/Documents/LEARNING/DataScience/ML/Recommender_System/ml-100k/ml-100k/u.user', sep='|', names=u_cols,encoding='latin-1')

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('C:/Users/a.kumar/Documents/LEARNING/DataScience/ML/Recommender_System/ml-100k/ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('C:/Users/a.kumar/Documents/LEARNING/DataScience/ML/Recommender_System/ml-100k/ml-100k/u.item', sep='|', names=i_cols,encoding='latin-1')

# split the ratings data in training and test category
ratings_train = pd.read_csv('C:/Users/a.kumar/Documents/LEARNING/DataScience/ML/Recommender_System/ml-100k/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('C:/Users/a.kumar/Documents/LEARNING/DataScience/ML/Recommender_System/ml-100k/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

# preparation to use user-user and item-item collaborative filtering
n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]

# create a user-item matrix to calculate the similarities between users and items
data_matrix=np.zeros((n_users,n_items))
for line in ratings.itertuples():
   data_matrix[line[1]-1, line[2]-1] = line[3]
   
data_matrix_using_pivot=ratings.pivot_table(index='user_id',columns='movie_id',values='rating')

# use the pairwise_distance function from sklearn to calculate the cosine similarity between two records
from sklearn.metrics.pairwise import pairwise_distances 
user_similarity = pairwise_distances(data_matrix, metric='cosine')
item_similarity = pairwise_distances(data_matrix.T, metric='cosine')

#Prediction function based on similarities
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #We use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

user_prediction = predict(data_matrix, user_similarity, type='user')
item_prediction = predict(data_matrix, item_similarity, type='item')

# using data_matrix_using_pivot table dataframe
ratings_mean=ratings.groupby('movie_id')['rating'].mean().sort_values(ascending=False)
ratings_count=ratings.groupby('movie_id')['rating'].count().sort_values(ascending=False)

ratings_mean_count = pd.DataFrame(ratings_mean)  
ratings_mean_count['ratings_count'] = pd.DataFrame(ratings_count)  

import matplotlib.pyplot as plt  
import seaborn as sns  
sns.set_style('dark')  

plt.figure(figsize=(8,6))  
plt.rcParams['patch.force_edgecolor'] = True  
ratings_mean_count['ratings_count'].hist(bins=50)  

plt.figure(figsize=(8,6))  
plt.rcParams['patch.force_edgecolor'] = True  
ratings_mean_count['rating'].hist(bins=50)  


plt.figure(figsize=(8,6))  
plt.rcParams['patch.force_edgecolor'] = True  
sns.jointplot(x='rating', y='ratings_count', data=ratings_mean_count, alpha=0.4)

movie_id_1=data_matrix_using_pivot[1]

movie_id_1_like_movies = data_matrix_using_pivot.corrwith(movie_id_1)

corr_movie_id_1 = pd.DataFrame(movie_id_1_like_movies, columns=['Correlation'])  
corr_movie_id_1.dropna(inplace=True)  
corr_movie_id_1.sort_values('Correlation', ascending=False).head(10)  
corr_movie_id_1 = corr_movie_id_1.join(ratings_mean_count['ratings_count']) 
corr_movie_id_1 = corr_movie_id_1.join(items['movie title'])  
movie_id_1_like_top_movies=corr_movie_id_1[corr_movie_id_1 ['ratings_count']>50].sort_values('Correlation', ascending=False).head()  

items.iloc[movie_id_1_like_top_movies.index,:]



# Recommendation using turicreate 
import turicreate
train_data = turicreate.SFrame(ratings_train)
test_data = turicreate.Sframe(ratings_test)

# Popular Model
popularity_model = turicreate.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')
popularity_recomm = popularity_model.recommend(users=[1,2,3,4,5],k=5)
popularity_recomm.print_rows(num_rows=25)


# Collaborative filtering model
item_sim_model = turicreate.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='cosine')

#Making recommendations
item_sim_recomm = item_sim_model.recommend(users=[1,2,3,4,5],k=5)
item_sim_recomm.print_rows(num_rows=25)


# using graphlab
import graphlab
train_data = graphlab.SFrame(ratings_train)
test_data = graphlab.SFrame(ratings_test)

# Popularity model
popularity_model = graphlab.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')
popularity_recomm = popularity_model.recommend(users=range(1,6),k=5)
popularity_recomm.print_rows(num_rows=25)


#Collaborative model
item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='pearson')

#Make Recommendations:
item_sim_recomm = item_sim_model.recommend(users=range(1,6),k=5)
item_sim_recomm.print_rows(num_rows=25)

# Evaluating model performance
model_performance = graphlab.compare(test_data, [popularity_model, item_sim_model])
graphlab.show_comparison(model_performance,[popularity_model, item_sim_model])




