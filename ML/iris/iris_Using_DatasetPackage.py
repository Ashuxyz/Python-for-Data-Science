# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:40:26 2019

@author: A.Kumar
"""

# iris data using dataset package
# load the iris dataset as an example
# when dataset is used to import a dataset, it has 4 parameters - data,target,feature_names
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



iris = load_iris()

# store the feature matrix (X) and response vector (y)

# uppercase X because it's an m x n matrix
X = iris.data

# lowercase y because it's a m x 1 vector
y = iris.target

# check the shapes of X and y
print('X dimensionality', X.shape)
print('y dimensionality', y.shape)

# examine the first 5 rows of the feature matrix (including the feature names)
data = pd.DataFrame(X, columns=iris.feature_names)

# examine the response vector
# this is a classification problem where you've 3 categories 0, 1, and 2
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=8)



# using Knn
k_range=range(1,26)
scores=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)

# fit the model with data (occurs in-place)
    knn.fit(X_train, y_train)
    pred_knn=knn.predict(X_test)
    scores.append(accuracy_score(y_test,pred_knn))

print('for k',scores.index(max(scores)),'score is',max(scores))
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))

# plot the relationship between K and testing accuracy
# plt.plot(x_axis, y_axis)
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()

# using GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
pred_gnb=gnb.predict(X_test)

print(accuracy_score(y_test, pred_gnb))
print(confusion_matrix(y_test, pred_gnb))
print(classification_report(y_test, pred_gnb))

# using Logistic Regression
logreg=LogisticRegression()
logreg.fit(X,y)
pred_logreg_full=logreg.predict(X)

print(accuracy_score(y, pred_logreg_full))
print(confusion_matrix(y, pred_logreg_full))
print(classification_report(y, pred_logreg_full))


logreg.fit(X_train,y_train)
pred_logreg_split=logreg.predict(X_test)

print(accuracy_score(y_test, pred_logreg_split))
print(confusion_matrix(y_test, pred_logreg_split))
print(classification_report(y_test, pred_logreg_split))

# Using AdaBoostClassifier

abc=AdaBoostClassifier()
abc.fit(X_train,y_train)
pred_abc=abc.predict(X_test)

print(accuracy_score(y_test, pred_abc))
print(confusion_matrix(y_test, pred_abc))
print(classification_report(y_test, pred_abc))


# using DecisionTreeClassifier

dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
pred_dtc=dtc.predict(X_test)

print(accuracy_score(y_test, pred_dtc))
print(confusion_matrix(y_test, pred_dtc))
print(classification_report(y_test, pred_dtc))


# Define the parameter values that should be searched
sample_split_range = list(range(2, 50))

# Create a parameter grid: map the parameter names to the values that should be searched
# Simply a python dictionary
# Key: parameter name
# Value: list of values that should be searched for that parameter
# Single key-value pair for param_grid
param_grid = dict(min_samples_split=sample_split_range)

# instantiate the grid
grid = GridSearchCV(dtc, param_grid, cv=10, scoring='accuracy')

# fit the grid with data
grid.fit(X_train, y_train)

# examine the best model

# Single best score achieved across all params (min_samples_split)
print(grid.best_score_)

# Dictionary containing the parameters (min_samples_split) used to generate that score
print(grid.best_params_)

# Actual model object fit with those best parameters
# Shows default parameters that we did not specify
print(grid.best_estimator_)

# using SVM

# Default kernel='rbf'
# We can change to others
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)

print(accuracy_score(y_test, pred_svm))
print(confusion_matrix(y_test, pred_svm))
print(classification_report(y_test, pred_svm))

# using K means

kmc = KMeans(n_clusters=3, max_iter=1000, n_init=20)
kmc.fit(X_train, y_train)
pred_kmc=kmc.predict(X_test)

print(accuracy_score(y_test, pred_kmc))
print(confusion_matrix(y_test, pred_kmc))
print(classification_report(y_test, pred_kmc))


# implementation of PCA for iris dataset

X_std = StandardScaler().fit_transform(X)

# Instantiate
pca = PCA(n_components=2)

# Fit and Apply dimensionality reduction on X
pca.fit_transform(X_std)

# Where the eigenvalues live
# You know first component and second component 
# has a and b percent of the data respectively
pca.explained_variance_ratio_

# Access components
pc_1 = pca.components_[0]
print(pc_1)
pc_2 = pca.components_[1]
print(pc_2)


# verifying the cross validation accuracy

# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
# k = 5 for KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

# Use cross_val_score function
# We are passing the entirety of X and y, not X_train or y_train, it takes care of splitting the dat
# cv=10 for 10 folds
# scoring='accuracy' for evaluation metric - althought they are many
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores)

# To select the best value of k for KNN model to predict species within cross validation

# search for an optimal value of K for KNN

# list of integers 1 to 30
# integers we want to try
k_range = range(1, 31)

# list of scores from k_range
k_scores = []

# 1. we will loop through reasonable values of k
for k in k_range:
    # 2. run KNeighborsClassifier with k neighbours
    knn = KNeighborsClassifier(n_neighbors=k)
    # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    # 4. append mean of scores for k neighbors to k_scores list
    k_scores.append(scores.mean())
print('for k',k_scores.index(max(k_scores)),'score is',max(k_scores))

# Hold a set from dataset and use it for cross validated model to get better prediction

k_hold_scores=[]

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    hold_scores=cross_val_score(knn,X_train,y_train,cv=10,scoring='accuracy')
    k_hold_scores.append(hold_scores.mean())
print('for k',k_hold_scores.index(max(k_hold_scores)),'score is',max(k_hold_scores))

knn.fit(X_train,y_train)
pred_after_cv=knn.predict(X_test)


#More efficient parameter tuning using GridSearchCV

k_range = list(range(1, 31))

# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range)
print(param_grid)

# instantiate the grid, here param_grid is used as range till 30 to repeat the 
#cross validation for each value
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')

grid.fit(X, y)
# view the complete results (list of named tuples), here mean is 
#the mean of accuracy for each fold
grid.grid_scores_



