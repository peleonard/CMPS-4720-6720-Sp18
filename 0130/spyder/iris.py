#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 22:02:33 2018

@author: LZQ
"""

##Pre-Analysis
import pandas as pd
#import numpy as np

##Acquire Data
iris = pd.read_csv("../iris.csv", 
                   names = ['sepLen', 'sepWid', 'petLen', 'petWid', 'class'])
iris



##Use 30/70 test/train split and confirm by .shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        iris.iloc[:, :4], iris['class'], test_size = .3)
print("X_train shape: {}".format(X_train.shape))


## EDA of variables
plot_sm = pd.plotting.scatter_matrix(X_train, alpha=.5, figsize=(8,8))


## Fitting and evaluating KNN algorithm, neighbors = 5
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Test predictions:\n {}".format(y_pred))
#print("Test score: {:.3f}".format(np.mean(y_pred == y_test)))
print("Test score: {:.3f}".format(knn.score(X_test, y_test)))
