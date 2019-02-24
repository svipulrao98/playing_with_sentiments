#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 19:30:35 2019

@author: vipss
"""

# importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
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

# converting words into vector and finding its tfidf value by using TfidfVectorizer
vect = TfidfVectorizer(min_df = 5).fit(X)
X_vectorized = vect.transform(X)

#train model(logistic)
logistic_classifier = LogisticRegression()
# fit data in the model
logistic_classifier.fit(X_vectorized, y)

#random forest Classification
forest_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
#fit to the model
forest_classifier.fit(X_vectorized, y)

#decision tree classification
tree_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
#train
tree_classifier.fit(X_vectorized, y)

#KNN
knn_classifier = KNeighborsClassifier(n_neighbors = 13, metric = 'minkowski', p = 2)

#train
knn_classifier.fit(X_vectorized, y)
