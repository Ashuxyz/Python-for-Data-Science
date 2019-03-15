# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:58:17 2019

@author: A.Kumar
"""

# Recommender system for movie lense

import pandas as pd

path1='C:/Users/a.kumar/Documents/LEARNING/DataScience/ML/Recommender_System/ml-latest-small/movies.csv'
path2='C:/Users/a.kumar/Documents/LEARNING/DataScience/ML/Recommender_System/ml-latest-small/ratings.csv'

movies=pd.read_csv(path1)
ratings=pd.read_csv(path2)

df=pd.merge(movies,ratings,on='movieId')

df.groupby('title')['rating'].mean().sort_values(ascending=False).head()

df.groupby('title')['rating'].count().sort_values(ascending=False).head() 

ratings_mean_count = pd.DataFrame(df.groupby('title')['rating'].mean())  

ratings_mean_count['rating_counts'] = pd.DataFrame(df.groupby('title')['rating'].count())  


import matplotlib.pyplot as plt  
import seaborn as sns  
sns.set_style('dark')  

plt.figure(figsize=(8,6))  
#plt.rcParams['patch.force_edgecolor'] = True  
ratings_mean_count['rating_counts'].hist(bins=50)  

plt.figure(figsize=(8,6))  
#plt.rcParams['patch.force_edgecolor'] = True  
ratings_mean_count['rating'].hist(bins=50)  


plt.figure(figsize=(8,6))  
plt.rcParams['patch.force_edgecolor'] = True  
sns.jointplot(x='rating', y='rating_counts', data=ratings_mean_count, alpha=0.4)

user_movie_rating = df.pivot_table(index='userId', columns='title', values='rating')    

forrest_gump_ratings = user_movie_rating['Forrest Gump (1994)']  

movies_like_forest_gump = user_movie_rating.corrwith(forrest_gump_ratings)

corr_forrest_gump = pd.DataFrame(movies_like_forest_gump, columns=['Correlation'])  
corr_forrest_gump.dropna(inplace=True)  
corr_forrest_gump.head()  

corr_forrest_gump.sort_values('Correlation', ascending=False).head(10)  


corr_forrest_gump = corr_forrest_gump.join(ratings_mean_count['rating_counts'])  
corr_forrest_gump.head()  


corr_forrest_gump[corr_forrest_gump ['rating_counts']>50].sort_values('Correlation', ascending=False).head()  
