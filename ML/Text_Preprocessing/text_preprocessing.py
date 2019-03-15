# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:19:05 2019

@author: A.Kumar
"""

import nltk
from nltk.tokenize import word_tokenize
# function to split text in to words
tokens=word_tokenize('Now I am bit more confident with Machine Learning and Deep Learning. Hope to keep it continue and will keep my consistency on.')
nltk.download('stopwords')
print(tokens)




from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
tokens=[w for w in tokens if not w in stop_words]
print(tokens)


#NLTK provides several stemmer interfaces like Porter stemmer, #Lancaster Stemmer, Snowball Stemmer
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
stems = []
for t in tokens:    
    stems.append(porter.stem(t))
print(stems)




