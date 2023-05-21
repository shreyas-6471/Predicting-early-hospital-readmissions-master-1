# -*- coding: utf-8 -*-
"""04_naive_bayes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12iFv4itHcdCG0JjfN135iSFqEElue8V-

# Naive Bayes:

Classifying with supervised learning whether diabetic patients are admitted, and if they are, if it's before or after 30 days.

Using the dataset from here: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
"""

# Commented out IPython magic to ensure Python compatibility.
from imblearn.over_sampling import SMOTE  # SMOTE oversampling
from imblearn.under_sampling import RandomUnderSampler  # Undersampling
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

# %pylab inline

# %config InlineBackend.figure_format = 'svg'
sns.set_style("white")

with open("y_train_liv.pkl", 'rb') as picklefile:
    y_train_liv = pickle.load(picklefile)

with open("y_test_liv.pkl", 'rb') as picklefile:
    y_test_liv = pickle.load(picklefile)

with open("x_train_liv.pkl", 'rb') as picklefile:
    x_train_liv = pickle.load(picklefile)

with open("x_test_liv.pkl", 'rb') as picklefile:
    x_test_liv = pickle.load(picklefile)

x_train = x_train_liv
y_train = y_train_liv
x_test = x_test_liv
y_test = y_test_liv

def makematrix(y_test, y_pred):
    score = accuracy_score(y_test, y_pred)

    confm = confusion_matrix(y_test, y_pred, labels=list(y_test.unique()))
    confm = confm.astype('float') / confm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10))
    sns.heatmap(confm, annot=True, fmt=".3f",
                linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size=15)

def pr_curve(y_test, model):
    y_test_num = y_test.copy()
    y_test_num = y_test_num.replace('NO', 0)
    y_test_num = y_test_num.replace('<30', 1)

    y_score = model.predict_proba(x_test)[:, 1]
    p, r, t = precision_recall_curve(y_test_num, y_score)

    # adding last threshold of 1. to threshold list
    t = np.concatenate((t, np.array([1.])))

    plt.plot(t, p, label='precision')
    plt.plot(t, r, label='recall')
    plt.title('Precision Recall Curve')
    plt.legend()

"""## Reducing the classes to binary classes ('>30' is now also 'NO'):"""

y_test = y_test.str.replace('>30', 'NO')
y_train = y_train.str.replace('>30', 'NO')

y_test.value_counts()

"""## Gaussian Naive Bayes:"""

NBmodel = naive_bayes.GaussianNB()
NBmodel.fit(x_train, y_train)
y_pred = NBmodel.predict(x_test)

print(classification_report(y_test, y_pred))

# makematrix(y_test, y_pred)

"""## Gaussian NB basically just predicted all <30, so recall for that class was great, but everything else was awful

## Bernoulli Naive Bayes:
"""

NBmodel = naive_bayes.BernoulliNB()
NBmodel.fit(x_train, y_train)
y_pred = NBmodel.predict(x_test)

print(classification_report(y_test, y_pred))

# makematrix(y_test, y_pred)

"""## I suppose this makes sense, because most of my data is categorical. I was really hoping for better performance from Gaussian, though.

## Multinomial Naive Bayes:
"""

NBmodel = naive_bayes.MultinomialNB()
NBmodel.fit(x_train, y_train)
y_pred = NBmodel.predict(x_test)

print(classification_report(y_test, y_pred))

# makematrix(y_test, y_pred)

"""## SMOTE:"""

sm = SMOTE(random_state=42)
x_train_smote, y_train_smote = sm.fit_resample(x_train, y_train)

"""## Multinomial + SMOTE:"""

NBmodel = naive_bayes.MultinomialNB()
NBmodel.fit(x_train_smote, y_train_smote)
y_pred = NBmodel.predict(x_test)

print(classification_report(y_test, y_pred))

# makematrix(y_test, y_pred)

"""## Gaussian + SMOTE:"""

NBmodel = naive_bayes.GaussianNB()
NBmodel.fit(x_train_smote, y_train_smote)
y_pred = NBmodel.predict(x_test)

print(classification_report(y_test, y_pred))

# makematrix(y_test, y_pred)

"""## Bernoulli + SMOTE:"""

NBmodel = naive_bayes.BernoulliNB()
NBmodel.fit(x_train_smote, y_train_smote)
y_pred = NBmodel.predict(x_test)

print(classification_report(y_test, y_pred))

# makematrix(y_test, y_pred)

"""## SMOTE did worse across the board.

## Gaussian NB with SMOTE also basically just predicted all <30, so recall for that class was great, but everything else was awful

## Undersampling:
"""

rus = RandomUnderSampler(random_state=0)
x_train_undersampled, y_train_undersampled = rus.fit_resample(x_train, y_train)

# sns.countplot(y_train_undersampled);

"""## Multinomial + Undersampling:"""

NBmodel = naive_bayes.MultinomialNB()
NBmodel.fit(x_train_undersampled, y_train_undersampled)
y_pred = NBmodel.predict(x_test)

print(classification_report(y_test, y_pred))

# makematrix(y_test, y_pred)

"""## Gaussian + Undersampling:"""

NBmodel = naive_bayes.GaussianNB()
NBmodel.fit(x_train_undersampled, y_train_undersampled)
y_pred = NBmodel.predict(x_test)

print(classification_report(y_test, y_pred))

# makematrix(y_test, y_pred)

"""## Bernoulli + Undersampling:"""

NBmodel = naive_bayes.BernoulliNB()
NBmodel.fit(x_train_undersampled, y_train_undersampled)
y_pred = NBmodel.predict(x_test)

print(classification_report(y_test, y_pred))

# makematrix(y_test, y_pred)

"""## Presumably, undersampling and oversampling hurt performance because part of the naive bayes calculation is the probability of each class occurring

## PR Curve for best model:
"""

NBmodel = naive_bayes.MultinomialNB()
NBmodel.fit(x_train, y_train)
pr_curve(y_test, model=NBmodel)

