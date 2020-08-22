# Heart-Disease-Prediction-Project
Heart Disease detection algorithm based upon the Kaggle Dataset on Heart Disease UCI.
# Parameters
There are a total of 14 columns, the columns are described as followed:

* age
* sex
* chest pain type (4 values)
* resting blood pressure
* serum cholestoral in mg/dl
* fasting blood sugar > 120 mg/dl
* resting electrocardiographic results (values 0,1,2)
* maximum heart rate achieved
* exercise induced angina
* oldpeak = ST depression induced by exercise relative to rest
* the slope of the peak exercise ST segment
* number of major vessels (0-3) colored by flourosopy
* thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
* target, i.e. whether the patient has heart diseases or not [0 for patient who has heart diseases & 1 for no heart diseases]
# Visualization
After seeing the heatmap it was evident that there is not much correlation between the independent variables so no feature scaling is needed
# Hyperparameters used for the model
param=[{'penalty':['l1','l2','elasticnet','none'],'C':np.logspace(-4,0.15,4,20),<br />
'solver':['lbfgs','newton-cg','liblinear','sag','saga'],'max_iter':[90,1000,2500,5000]}]<br />
clf=GridSearchCV(classifier,param_grid=param,cv=5,verbose=True,n_jobs=-1)<br />
best_clf=clf.fit(X_train,y_train)<br />
best_parameters=best_clf.best_estimator_<br />

# Final Model
For a test model I have used Random Forest Classifier and measured the accuracy on the basis of Roc_Auc score.<br />
classifier =LogisticRegression(C=0.0024173154808041063, class_weight=None, dual=False,<br />
                   fit_intercept=True, intercept_scaling=1,max_iter=90, multi_class='auto', n_jobs=None, penalty='l2',<br />
                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,<br />
                   warm_start=False)<br />
# Results                   
 The model has given an accuracy of 86.33% over the test dataset that is randomly generated by 20% of the main datset.                  
                   
                   
