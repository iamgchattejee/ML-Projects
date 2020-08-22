# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:00:24 2020

@author: Gaurav
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('heart.csv')
X = dataset.iloc[:,1:13].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc1= StandardScaler()
X_train=sc1.fit_transform(X_train)
X_test=sc1.transform(X_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier =LogisticRegression(C=0.0024173154808041063, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1,max_iter=90, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
classifier.fit(X_train, y_train)

#HyperParameters Tweking
param=[{'penalty':['l1','l2','elasticnet','none'],'C':np.logspace(-4,4,20),
'solver':['lbfgs','newton-cg','liblinear','sag','saga'],'max_iter':[100,1000,2500,5000]}]

from sklearn.model_selection import GridSearchCV

clf=GridSearchCV(classifier,param_grid=param,cv=2,verbrose=True,n_jobs=-1)
best_clf=clf.fit(X_train,y_train)
best_parameters=best_clf.best_estimator_

# Predicting a new result
y_predict=classifier.predict(X_test)


from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_predict)
print(cm)

from sklearn.metrics import roc_auc_score
roc=roc_auc_score(y_test, y_predict)

