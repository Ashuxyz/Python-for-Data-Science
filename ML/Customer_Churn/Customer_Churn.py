# Customer churn prediction

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score
from sklearn.linear_model import LogisticRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

datafile='C:/Users/a.kumar/Documents/LEARNING/DataScience/Customer_Churn/Customer_Churn.csv'
df=pd.read_csv(datafile)

df.columns=[col.replace(' ','_').lower() for col in df.columns]

df['churn']= df['churn?'].map({'False.':0, 'True.':1})

df['intl_plan']=df["int'l_plan"].map({'no':0, 'yes':1})
df['voicemail_plan']=df['vmail_plan'].map({'no':0, 'yes':1})

Nan_col_list=[]
for col in df.columns:
    count= df[col].isnull().sum()
    if count!=0:
        Nan_col_list.append(col)
        
print(Nan_col_list)


#decide relationship between target and each feature to select the 
#columns to be used for model
corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
heat_map=plt.gcf()
heat_map.set_size_inches(20,15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# Establish a Baseline
y_count=df['churn'].size
y_0_count=df['churn'].value_counts()[0]
y_1_count=df['churn'].value_counts()[1]
y_0_count_percentage=y_0_count*100/y_count
y_1_count_percentage=y_1_count*100/y_count

print(y_0_count_percentage,y_1_count_percentage)

selected_features=['account_length','area_code','intl_plan','voicemail_plan','vmail_message','day_mins','day_calls','day_charge','eve_mins','eve_calls','eve_charge','night_mins','night_calls','night_charge','intl_mins','intl_calls','intl_charge','custserv_calls']

X_df=df[selected_features]
X=np.array(X_df)
y=np.array(df['churn'])

seed=21
scoring='accuracy'

# Dataset split function
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=seed)

logreg=LogisticRegression()
logreg.fit(X_train,y_train)
rr = Ridge(alpha=0.01)
rr.fit(X_train, y_train)

rr100 = Ridge(alpha=100)
rr100.fit(X_train, y_train)

train_score=logreg.score(X_train, y_train)
test_score=logreg.score(X_test, y_test)
Ridge_train_score = rr.score(X_train,y_train)
Ridge_test_score = rr.score(X_test, y_test)
Ridge_train_score100 = rr100.score(X_train,y_train)
Ridge_test_score100 = rr100.score(X_test, y_test)



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


# KNN Hyperparamater tuning
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

# Hyper tuning Logistic Regression using GridSearchCV
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg,param_grid,cv=5)

# Fit it to the training data
logreg_cv.fit(X_train,y_train)

# Print the optimal parameters and best score
print('Tuned Logistic Regression Parameter: {}'.format(logreg_cv.best_params_))
print('Tuned Logistic Regression Accuracy: {}'.format(logreg_cv.best_score_))


# Aplying upsampling to improve the performance
df_majority = df[df['churn']==0]
df_minority = df[df['churn']==1]

df_minority_upsampled = resample(df_minority,replace=True,
n_samples=2850, #same number of samples as majority classe
random_state=seed) #set the seed for random resampling
# Combine resampled results
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

df_upsampled['churn'].value_counts()

X_df_upsampled=df_upsampled[selected_features]
X_upsampled=np.array(X_df_upsampled)
y_upsampled=np.array(df_upsampled['churn'])

X_train_upsampled,X_test_upsampled,y_train_upsampled,y_test_upsampled= train_test_split(X_upsampled,y_upsampled,test_size=0.25,random_state=seed)

results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=10, random_state=seed)
	cv_results = cross_val_score(model, X_train_upsampled, y_train_upsampled, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

def score_dataset(algorithm,X_train_upsampled, X_test_upsampled, y_train_upsampled, y_test_upsampled):
    model = algorithm
    model.fit(X_train_upsampled, y_train_upsampled)
    preds = model.predict(X_test_upsampled)
 #   preds_proba = model.predict_proba(X_test)
 #   preds_proba=[p[1] for p in preds_proba]
    print(accuracy_score(y_test_upsampled, preds))
    print(confusion_matrix(y_test_upsampled, preds))
    print(classification_report(y_test_upsampled, preds))
  #  print('Unbalanced model AUROC: ' + str(roc_auc_score(y_test, preds_proba)))

for name, model in models:
    print('Performance Report for Algorithm',name,'is as below')
    score_dataset(model,X_train_upsampled,X_test_upsampled,y_train_upsampled,y_test_upsampled)


# KNN Hyperparamater tuning
k_range=range(1,26)
scores=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)

# fit the model with data (occurs in-place)
    knn.fit(X_train_upsampled, y_train_upsampled)
    pred_knn=knn.predict(X_test_upsampled)
    scores.append(accuracy_score(y_test_upsampled,pred_knn))

print('for k',scores.index(max(scores)),'score is',max(scores))
print(confusion_matrix(y_test_upsampled, pred_knn))
print(classification_report(y_test_upsampled, pred_knn))

# Hyper tuning Logistic Regression using GridSearchCV
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg,param_grid,cv=5)

# Fit it to the training data
logreg_cv.fit(X_train_upsampled,y_train_upsampled)

# Print the optimal parameters and best score
print('Tuned Logistic Regression Parameter: {}'.format(logreg_cv.best_params_))
print('Tuned Logistic Regression Accuracy: {}'.format(logreg_cv.best_score_))

#using Adaboost Classifier

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=200)
bdt.fit(X_train,y_train)
preds_ada=bdt.predict(X_test)
print(accuracy_score(y_test, preds_ada))
print(confusion_matrix(y_test,preds_ada))
print(classification_report(y_test, preds_ada))
