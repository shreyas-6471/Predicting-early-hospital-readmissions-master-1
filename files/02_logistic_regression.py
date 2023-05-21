# -*- coding: utf-8 -*-
"""02_logistic_regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iG9FD4P-nsbqU9XpnlpUq-R-JCOO5Dbw

# Logistic Regression:

Classifying with supervised learning whether diabetic patients are readmitted, and if they are, if it's before or after 30 days.

Using the dataset from here: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# SMOTE
from imblearn.over_sampling import SMOTE

# Undersampling
from imblearn.under_sampling import RandomUnderSampler

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import pickle

# Using scaled X_train and X_test because we will want to try regularization

with open("y_train_liv.pkl", 'rb') as picklefile: 
    y_train = pickle.load(picklefile)

with open("y_test_liv.pkl", 'rb') as picklefile: 
    y_test = pickle.load(picklefile)

with open("x_train_scaled_liv.pkl", 'rb') as picklefile: 
    x_train = pickle.load(picklefile)

with open("x_test_scaled_liv.pkl", 'rb') as picklefile: 
    x_test = pickle.load(picklefile)

"""## Converting to binary classes:"""

y_test = y_test.str.replace('>30','NO')
y_train = y_train.str.replace('>30','NO')

"""## Logistic regression with balanced class weights with test/train split (25% for test):"""

lrmodel = linear_model.LogisticRegression(class_weight="balanced")
lrmodel.fit(x_train, y_train)

# Predict on test
y_pred = lrmodel.predict(x_test)

# Score on test
score = metrics.accuracy_score(y_test, y_pred)
#print(score)
print("Accuracy: %.3f"% score)
print(metrics.classification_report(y_test, y_pred))

"""## Logistic regression with cross-validation:

> Indented block


"""

# 10-fold cross-validation with logistic regression
# stratifying the Kfold splits is default in CV
# returning the average f1_macro score
print(cross_val_score(lrmodel, x_train, y_train, cv=5, scoring='f1_macro').mean())

y_test.unique()

cm = metrics.confusion_matrix(y_test, y_pred, labels=list(y_test.unique()))
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm

print("Accuracy: %.3f"% metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

lrmodel.coef_.shape

"""## Get the coefficients:"""

coefficients = pd.DataFrame({"Feature":x_train.columns,"Coefficients":np.transpose(lrmodel.coef_[0,])})

coefficients['abs_val_coef'] = coefficients.Coefficients.abs()

coefficients.sort_values(by=['abs_val_coef'], ascending=False).head(10)

"""## Grid search Logistic Regression:"""

# list(np.arange(0.0, 10.0, 0.1))

# define the parameter values that should be searched
C_range = list(np.arange(0.1, 5.2, 0.2))
print(f"testing values of C: {C_range}")

# Logistic regression defaults to L2 normalization

# create a parameter grid: map the parameter names to the values that should be searched 
param_grid = dict(C=C_range)

# instantiate the grid
grid = GridSearchCV(lrmodel, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)

# fit the grid with data 
grid.fit(x_train, y_train);

# view the complete results (list of named tuples)
# grid.grid_scores_ # old name for this function
grid.cv_results_

# examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

"""## Plot precision-recall curve:"""

from sklearn.metrics import precision_recall_curve

y_test_num = y_test.copy()
y_test_num = y_test_num.replace('NO', 0)
y_test_num = y_test_num.replace('<30', 1)

model = lrmodel

y_score = model.predict_proba(x_test)[:, 1]
p, r, t = precision_recall_curve(y_test_num, y_score)
# adding last threshold of 1. to threshold list
t = np.concatenate((t, np.array([1.])))

plt.plot(t, p, label='precision')
plt.plot(t, r, label='recall')
plt.title('Precision Recall Curve')
plt.legend()

"""## Random undersampling:"""

rus = RandomUnderSampler(random_state=0)
x_train_undersampled, y_train_undersampled = rus.fit_sample(x_train, y_train)

lrmodel.fit(x_train_undersampled, y_train_undersampled)

# Predict on test
y_pred = lrmodel.predict(x_test)
print(metrics.classification_report(y_test, y_pred))

"""## Grid search tuning of C for random undersampling:"""

# define the parameter values that should be searched
C_range = list(np.arange(0.1, 5.2, 0.2))
print(f"testing values of C: {C_range}")

# Logistic regression defaults to L2 normalization

# create a parameter grid: map the parameter names to the values that should be searched 
param_grid = dict(C=C_range)

# instantiate the grid
grid = GridSearchCV(lrmodel, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)

# fit the grid with data 
grid.fit(x_train_undersampled, y_train_undersampled);

# examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

lrmodel = grid.best_estimator_
lrmodel.fit(x_train_undersampled, y_train_undersampled)
# Predict on test
y_pred = lrmodel.predict(x_test)

print(metrics.classification_report(y_test, y_pred))

"""## Get the coefficients:"""

coefficients = pd.DataFrame({"Feature":x_train.columns,"Coefficients":np.transpose(lrmodel.coef_[0,])})

coefficients['abs_val_coef'] = coefficients.Coefficients.abs()

coefficients.sort_values(by=['abs_val_coef'], ascending=False).head(10)

"""## Polynomial features:"""

poly = preprocessing.PolynomialFeatures(2, interaction_only=True)
x_train_poly_undersampled = poly.fit_transform(x_train_undersampled)

x_train_poly_undersampled.shape

x_test_poly_undersampled = poly.transform(x_test)

lrmodel = linear_model.LogisticRegression(C=0.01) # increasing regularization due to polynomial features

lrmodel.fit(x_train_poly_undersampled, y_train_undersampled)

# Predict on test
y_pred = lrmodel.predict(x_test_poly_undersampled)

print(metrics.classification_report(y_test, y_pred))

"""## Get the coefficients:"""

coefficients = pd.DataFrame({"Feature":list(poly.get_feature_names(x_train.columns)),
                             "Coefficients":np.transpose(lrmodel.coef_[0,])})

coefficients['abs_val_coef'] = coefficients.Coefficients.abs()

coefficients.sort_values(by=['abs_val_coef'], ascending=False).head(10)

"""## Logistic with SMOTE:"""

sm = SMOTE(random_state=42)
x_train_smote, y_train_smote = sm.fit_sample(x_train, y_train)

# sns.countplot(y_test)

# sns.countplot(y_train_smote)

lrmodel = linear_model.LogisticRegression(C=1)
lrmodel.fit(x_train_smote, y_train_smote)

# Predict on test
y_pred = lrmodel.predict(x_test)
print(metrics.classification_report(y_test, y_pred))

"""## Get the coefficients:"""

coefficients = pd.DataFrame({"Feature":x_train.columns,"Coefficients":np.transpose(lrmodel.coef_[0,])})

coefficients['abs_val_coef'] = coefficients.Coefficients.abs()

coefficients.sort_values(by=['abs_val_coef'], ascending=False).head(10)

