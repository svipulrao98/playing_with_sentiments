#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 19:21:06 2019

@author: vipss
"""

import classifiers as cl

choice=1

while choice==1:

    s = input('Enter the string: ')
    
    s = cl.re.sub(r"http\S+", "", s)
    s = cl.re.sub(r"@\S+", "", s)
    
    pre = cl.logistic_classifier.predict(cl.vect.transform([s]))
    print('According to logistic Classifier:', pre[0])
    
    pre = cl.knn_classifier.predict(cl.vect.transform([s]))
    print('According to KNN Classifier:', pre[0])
    
    pre = cl.forest_classifier.predict(cl.vect.transform([s]))
    print('According to Random Forest Classifier:', pre[0])
    
    pre = cl.tree_classifier.predict(cl.vect.transform([s]))
    print('According to Decision Tree Classifier:', pre[0])
    
    choice = int(input('Test again?(enter 1 if yes, any other number if no):'))
