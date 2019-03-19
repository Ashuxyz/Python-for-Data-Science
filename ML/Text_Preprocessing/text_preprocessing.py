# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:19:05 2019

@author: A.Kumar
"""

import nltk
from nltk.tokenize import word_tokenize
# function to split text in to words
tokens=word_tokenize('Now I am bit more confident with Machine Learning and Deep Learning. Hope to keep it continue and will keep my consistency on.')

vocabulary = set(tokens)
print(len(vocabulary))
frequency_dist = nltk.FreqDist(tokens)
sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)

from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
tokens=[w for w in tokens if not w in stop_words]
print(tokens)


from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud().generate_from_frequencies(frequency_dist)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


#NLTK provides several stemmer interfaces like Porter stemmer, #Lancaster Stemmer, Snowball Stemmer
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
stems = []
for t in tokens:    
    stems.append(porter.stem(t))
print(stems)

#We use TfidTransformer to covert the text corpus into the feature vectors, 
#we restrict the maximum features to 10000.


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(tokens)
train_vectors.toarray()
test_vectors = vectorizer.transform(X_test)
print(train_vectors.shape, test_vectors.shape)


