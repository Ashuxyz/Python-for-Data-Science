# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:40:26 2019

@author: A.Kumar
"""

# Vectorization, Multinomial Naive Bayes Classifier and Evaluation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score
from sklearn.linear_model import LogisticRegression

# Representing text as Numerical

# example text for model training (SMS messages)
simple_train = ['call you tonight', 'Call me a cab', 'please call me.. please']

# 1. import and instantiate CountVectorizer (with the default parameters)
# 2. instantiate CountVectorizer (vectorizer)
vect = CountVectorizer()

# 3. fit
# learn the 'vocabulary' of the training data (occurs in-place)
vect.fit(simple_train)

# examine the fitted vocabulary
vect.get_feature_names()

# 4. transform training data into a 'document-term matrix' or Sparse matrix
simple_train_dtm = vect.transform(simple_train)

# convert sparse matrix to a dense matrix
simple_train_dtm.toarray()
pd.DataFrame(simple_train_dtm.toarray(),columns=vect.get_feature_names())

# example text for model testing
simple_test = ['Please don\'t call me']

# 4. transform testing data into a document-term matrix (using existing vocabulary)
simple_test_dtm = vect.transform(simple_test)
simple_test_dtm.toarray()

# examine the vocabulary and document-term matrix together
pd.DataFrame(simple_test_dtm.toarray(), columns=vect.get_feature_names())

# Reading a text based dataset into pandas
url='https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
sms=pd.read_table(url, header=None, names=['label', 'message'])

# examine the class distribution
sms.label.value_counts()

# convert label to a numerical variable
sms['label_num'] = sms.label.map({'ham':0, 'spam':1})

# how to define X and y (from the SMS data) for use with COUNTVECTORIZER. Here X is 1D 
#currently because it will be passed to Vectorizer to become a 2D matrix
#You must always have a 1D object so CountVectorizer can turn into a 2D object for 
#the model to be built on
X = sms.message
y = sms.label_num
print(X.shape)
print(y.shape)

# split X and y into training and testing sets
# by default, it splits 75% training and 25% test
# random_state=1 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Vectorizing the dataset
# 2. instantiate the vectorizer
vect = CountVectorizer()
vect.fit(X_train)
vect.get_feature_names()
#X_train_dtm=vect.transform(X_train)
## equivalently: combine fit and transform into a single step
# this is faster and what most people would do
X_train_dtm = vect.fit_transform(X_train)

# 4. transform testing data (using fitted vocabulary) into a document-term matrix. 
#Here prediction is not required
X_test_dtm = vect.transform(X_test)

X_train_dtm.toarray()
X_test_dtm.toarray()

pd.DataFrame(X_test_dtm.toarray(),columns=vect.get_feature_names())

# Building and evaluating a model
#  instantiate a Multinomial Naive Bayes model
nb = MultinomialNB()

# 3. train the model using vectorized training dataset
# using X_train_dtm (timing it with an IPython "magic command")

%time nb.fit(X_train_dtm, y_train)

# make class predictions for X_test_dtm
y_pred_class_nb = nb.predict(X_test_dtm)

accuracy_score(y_test,y_pred_class_nb)

classification_report(y_test,y_pred_class_nb)

# examine class distribution
print(y_test.value_counts())

# calculate null accuracy (for multi-class classification problems)
# .head(1) assesses the value 1208
null_accuracy = y_test.value_counts().head(1) / len(y_test)
print('Null accuracy:', null_accuracy)

# Manual calculation of null accuracy by always predicting the majority class
print('Manual null accuracy:',(1208 / (1208 + 185)))

# print the confusion matrix
confusion_matrix(y_test,y_pred_class_nb)

# print message text for the false positives (ham incorrectly classified as spam)
X_test[y_pred_class_nb > y_test]

# alternative less elegant but easier to understand
# X_test[(y_pred_class==1) & (y_test==0)]

# print message text for the false negatives (spam incorrectly classified as ham)
X_test[y_pred_class_nb < y_test]
# alternative less elegant but easier to understand
# X_test[(y_pred_class=0) & (y_test=1)]


# calculate predicted probabilities for X_test_dtm (poorly calibrated)

# Numpy Array with 2C
# left Column: probability class 0
# right C: probability class 1
# we only need the right column 
y_pred_prob_nb = nb.predict_proba(X_test_dtm)[:, 1]

# Naive Bayes predicts very extreme probabilites, you should not take them at face value

# calculate AUC
roc_auc_score(y_test, y_pred_prob_nb)

# comparing against different models

# Consider Logistic Regression

logreg=LogisticRegression()
%time logreg.fit(X_train_dtm, y_train)

# by seeing the wall time, it is much slower than Naive Bias

y_pred_class_logreg=logreg.predict(X_test_dtm)

y_pred_prob_logreg=logreg.predict_proba(X_test_dtm)[:,1]

accuracy_score(y_test,y_pred_class_logreg)
roc_auc_score(y_test,y_pred_prob_logreg)

#Examining the model for further insight
#We will examine the our trained Naive Bayes model to calculate the 
#approximate "spamminess" of each token.

# store the vocabulary of X_train
X_train_tokens = vect.get_feature_names()
len(X_train_tokens)

# Naive Bayes counts the number of times each token appears in each class
# trailing underscore - learned during fitting
nb.feature_count_

# rows represent classes, columns represent tokens
nb.feature_count_.shape

# number of times each token appears across all HAM messages
ham_token_count = nb.feature_count_[0, :]
ham_token_count


# number of times each token appears across all SPAM messages
spam_token_count = nb.feature_count_[1, :]
spam_token_count

# create a DataFrame of tokens with their separate ham and spam counts
tokens = pd.DataFrame({'token':X_train_tokens, 'ham':ham_token_count, 'spam':spam_token_count}).set_index('token')

# examine 5 random DataFrame rows
# random_state=6 is a seed for reproducibility
tokens.sample(5, random_state=6)

# Naive Bayes counts the number of observations in each class
nb.class_count_
nb.class_count_.shape

# add 1 to ham and spam counts to avoid dividing by 0
tokens['ham'] = tokens.ham + 1
tokens['spam'] = tokens.spam + 1
tokens.sample(5, random_state=6)

# convert the ham and spam counts into frequencies
tokens['ham'] = tokens.ham / nb.class_count_[0]
tokens['spam'] = tokens.spam / nb.class_count_[1]
tokens.sample(5, random_state=6)

# calculate the ratio of spam-to-ham for each token
tokens['spam_ratio'] = tokens.spam / tokens.ham
tokens.sample(5, random_state=6)

# examine the DataFrame sorted by spam_ratio, highest is more spamminess
tokens.sort_values('spam_ratio', ascending=False)

#Countvectorizer tuning with some more parameters

# remove English stop words. It is used to remove english word from the list.
vect1 = CountVectorizer(stop_words='english')























