#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:30:14 2019

@author: vipss
"""

# importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import re


#get data
dataset = pd.read_csv('text_emotion.csv')

#remove Nan values
dataset.replace([np.NaN],[''],inplace=True)

#store
X=dataset['content'].values

#remove urls
for i in range(len(X)):
    X[i]=re.sub(r"http\S+", "", X[i])

#remove all twitter usernamesfrom sklearn.svm import SVC

for i in range(len(X)):
    X[i]=re.sub(r"@\S+", "", X[i])

#categories
y=dataset['sentiment'].values

#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# converting words into vector and finding its tfidf value by using TfidfVectorizer
vect = TfidfVectorizer(min_df = 5).fit(X_train)
X_train_vectorized = vect.transform(X_train)
X_test_vectorized = vect.transform(X_test)

#train model(logistic)
logistic_classifier = LogisticRegression()
# fit data in the model
logistic_classifier.fit(X_train_vectorized, y_train)
# predict by using the model
predictions = logistic_classifier.predict(X_test_vectorized)

#present in form of confusion matrix
cm = confusion_matrix(y_test, predictions)

#random forest Classification
forest_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest_classifier.fit(X_train_vectorized, y_train)

#testing
predictions1 = forest_classifier.predict(vect.transform(X_test))

#present in form of confusion matrix
cm1 = confusion_matrix(y_test, predictions1)

#decision tree classification
tree_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

#train
tree_classifier.fit(X_train_vectorized, y_train)

#testing
predictions2 = tree_classifier.predict(X_test_vectorized)

#KNN
knn_classifier = KNeighborsClassifier(n_neighbors = 13, metric = 'minkowski', p = 2)

#train
knn_classifier.fit(X_train_vectorized, y_train)

#test
predictions3=knn_classifier.predict(X_test_vectorized)


#naive bayes
# =============================================================================
# POOREST
# =============================================================================
naive_classifier = GaussianNB()
#train
naive_classifier.fit(X_train_vectorized.toarray(), y_train)
#test
predictions4=naive_classifier.predict(X_test_vectorized.toarray())
