# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 14:44:25 2019

@author: A.Kumar
"""

# Recommender system for movies

import pandas as pd
path = 'C:/Users/a.kumar/Documents/LEARNING/DataScience/ML/Recommender_System/file.tsv'
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df=pd.read_csv(path,names=column_names,sep='\t')

# Check out all the movies and their respective IDs 
movie_titles = pd.read_csv('https://cdncontribute.geeksforgeeks.org/wp-content/uploads/Movie_Id_Titles.csv') 

movie_data=pd.merge(df,movie_titles,on='item_id')

movie_data['title'].value_counts()
movie_data['rating'].value_counts()

movie_with_highest_rating=movie_data[movie_data['rating']==5]
movie_with_highest_rating['title'].value_counts()

# Calculate mean rating of all movies 
movie_data.groupby('title')['rating'].mean().sort_values(ascending=False).head() 

# Calculate count rating of all movies 
movie_data.groupby('title')['rating'].count().sort_values(ascending=False).head() 

# creating dataframe with 'rating' count values 
ratings = pd.DataFrame(movie_data.groupby('title')['rating'].mean()) 

ratings['num of ratings'] = pd.DataFrame(movie_data.groupby('title')['rating'].count()) 

ratings.head() 


#visualization
import matplotlib.pyplot as plt 
import seaborn as sns 

sns.set_style('white') 

# plot graph of 'num of ratings column' 
plt.figure(figsize =(10, 4)) 

ratings['num of ratings'].hist(bins = 70) 

# plot graph of 'ratings' column 
plt.figure(figsize =(10, 4)) 

ratings['rating'].hist(bins = 70) 


# Sorting values according to 
# the 'num of rating column' 
moviemat = movie_data.pivot_table(index ='user_id', columns ='title', values ='rating') 

moviemat.head() 

ratings.sort_values('num of ratings', ascending = False).head(10) 

# analysing correlation with similar movies 
starwars_user_ratings = moviemat['Star Wars (1977)'] 
liarliar_user_ratings = moviemat['Liar Liar (1997)'] 

starwars_user_ratings.head() 

# analysing correlation with similar movies 
similar_to_starwars = moviemat.corrwith(starwars_user_ratings) 
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings) 

corr_starwars = pd.DataFrame(similar_to_starwars, columns =['Correlation']) 
corr_starwars.dropna(inplace = True) 

corr_starwars.head() 

# Similar movies like starwars 
corr_starwars.sort_values('Correlation', ascending = False).head(10) 
corr_starwars = corr_starwars.join(ratings['num of ratings']) 

corr_starwars.head() 

corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation', ascending = False).head() 


# Similar movies as of liarliar 
corr_liarliar = pd.DataFrame(similar_to_liarliar, columns =['Correlation']) 
corr_liarliar.dropna(inplace = True) 

corr_liarliar = corr_liarliar.join(ratings['num of ratings']) 
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation', ascending = False).head() 








