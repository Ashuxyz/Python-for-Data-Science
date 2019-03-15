# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:54:14 2019

@author: A.Kumar
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:50:33 2019

@author: A.Kumar
"""

import pandas as pd

simple_train = ['call you tonight', 'Call me a cab', 'please call me.. please']

# using countvectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(simple_train)
vect.get_feature_names()
print(vect.get_feature_names())
simple_train_dtm_vect = vect.transform(simple_train)
simple_train_dtm_vect.toarray()
pd.DataFrame(simple_train_dtm_vect.toarray(),columns=vect.get_feature_names())


# using Tokenizer
from keras.preprocessing.text import Tokenizer
max_words = 50
max_len = 20
tok = Tokenizer(num_words=max_words)
simple_train_dtm_tok=tok.fit_on_texts(simple_train)
sequences = tok.texts_to_sequences(simple_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

# Using nltk.word_tokenize
import nltk
from nltk.tokenize import word_tokenize
simple_train = ['call you tonight', 'Call me a cab', 'please call me.. please']

# function to split text in to words
list_words=[]
for i in simple_train:
    tokens=word_tokenize(i)
    for t in tokens:
        list_words.append(t)        
print(list_words)
    
