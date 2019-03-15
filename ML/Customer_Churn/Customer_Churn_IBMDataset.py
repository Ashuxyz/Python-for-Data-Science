# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 08:23:21 2019

@author: A.Kumar
"""

# Customer churn analysis for dataset provided by IBM

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample

datafile='C:/Users/a.kumar/Documents/LEARNING/DataScience/Customer_Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df=pd.read_csv(datafile)

df.columns=[col.replace(' ','_').lower() for col in df.columns]


df['gender_b']=df['gender'].map({'Female':1,'Male':0})
df['partner_b']=df.partner.map({'Yes':1,'No':0})
df['dependents_b']=df.dependents.map({'Yes':1,'No':0})
df['phoneservice_b']=df.phoneservice.map({'Yes':1,'No':0})
df['multiplelines_b']=df.multiplelines.map({'Yes':1,'No':0,'No phone service':0})
df['internetservice_b']=df.internetservice.map({'DSL':1,'Fiber optic':2,'No':0})
df['onlinesecurity_b']=df.onlinesecurity.map({'Yes':1,'No':0,'No internet service':0})
df['onlinebackup_b']=df.onlinebackup.map({'Yes':1,'No':0,'No internet service':0})
df['deviceprotection_b']=df.deviceprotection.map({'Yes':1,'No':0,'No internet service':0})
df['techsupport_b']=df.techsupport.map({'Yes':1,'No':0,'No internet service':0})
df['streamingtv_b']=df.streamingtv.map({'Yes':1,'No':0,'No internet service':0})
df['streamingmovies_b']=df.streamingmovies.map({'Yes':1,'No':0,'No internet service':0})
df['contract_b']=df.contract.map({'Month-to-month':1,'One year':2,'Two year':3})
df['paperlessbilling_b']=df.paperlessbilling.map({'Yes':1,'No':0})
df['paymentmethod_b']=df.paymentmethod.map({'Electronic check':1,'Mailed check':2,'Bank transfer (automatic)':3,'Credit card (automatic)':4})
df['churn_b']=df.churn.map({'Yes':1,'No':0})


#changing total charges from object to numeric
df['totalcharges']=pd.to_numeric(df['totalcharges'],errors='coerce')


#checking if any column is having Nan values
Nan_col_list=[]
for col in df.columns:
    count= df[col].isnull().sum()
    if count!=0:
        Nan_col_list.append(col)
        
print(Nan_col_list)

#replacing null value with 0 
df['totalcharges'].fillna(0,inplace=True)
    

#decide relationship between target and each feature to select the 
#columns to be used for model
corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
heat_map=plt.gcf()
heat_map.set_size_inches(20,15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

#selection of features to be used in the model
selected_features=['gender_b','seniorcitizen','partner_b','dependents_b','tenure','phoneservice_b','multiplelines_b','internetservice_b','onlinesecurity_b','onlinebackup_b','deviceprotection_b','techsupport_b','streamingtv_b','streamingmovies_b','contract_b','paperlessbilling_b','paymentmethod_b','monthlycharges','totalcharges']
selected_features_without_totalcharges=['gender_b','seniorcitizen','partner_b','dependents_b','tenure','phoneservice_b','multiplelines_b','internetservice_b','onlinesecurity_b','onlinebackup_b','deviceprotection_b','techsupport_b','streamingtv_b','streamingmovies_b','contract_b','paperlessbilling_b','paymentmethod_b','monthlycharges']

X_df_without=df[selected_features_without_totalcharges]
X_df_with=df[selected_features]
X=np.array(X_df_with)
y=np.array(df['churn_b'])


# establish a baseline
y_count=df['churn_b'].size
y_0_count=df['churn_b'].value_counts()[0]
y_1_count=df['churn_b'].value_counts()[1]

y_0_count_percentage=y_0_count*100/y_count
y_1_count_percentage=y_1_count*100/y_count

seed=20
scoring = 'accuracy'

# Dataset split function
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=seed)
pd.DataFrame(X_train).to_csv('X_train.csv')
pd.DataFrame(X_test).to_csv('X_test.csv')

# Evaluate multiple algorithm
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RF', RandomForestClassifier()))

results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=10, random_state=seed)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

def score_dataset(algorithm,X_train, X_test, y_train, y_test):
    model = algorithm
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
 #   preds_proba = model.predict_proba(X_test)
 #   preds_proba=[p[1] for p in preds_proba]
    print(accuracy_score(y_test, preds))
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
  #  print('Unbalanced model AUROC: ' + str(roc_auc_score(y_test, preds_proba)))

for name, model in models:
    print('Performance Report for Algorithm',name,'is as below')
    score_dataset(model,X_train,X_test,y_train,y_test)

# up-sampling the minority class
data_majority = df[df['churn_b']==0]
data_minority = df[df['churn_b']==1]

data_minority_upsampled = resample(data_minority,replace=True,
n_samples=5174, #same number of samples as majority classe
random_state=seed) #set the seed for random resampling
# Combine resampled results
data_upsampled = pd.concat([data_majority, data_minority_upsampled])

data_upsampled['churn_b'].value_counts()

# 

