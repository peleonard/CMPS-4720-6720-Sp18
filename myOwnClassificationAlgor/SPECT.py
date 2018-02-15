#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 18:56:42 2018

@author: LZQ
"""
#%%
#import the data SPECT
import pandas as pd
import numpy as np
#!cat SPECTtrain.txt
sp_names = ['F' + str(i) for i in range(23)]
sp_names[0] = 'diagnosis'
SP_train = pd.read_csv("SPECTtrain.txt", sep=",", names=sp_names)
SP_test = pd.read_csv("SPECTtest.txt", sep=",", names=sp_names)
SP_train.shape
SP_test.shape


#%%
def perceptronLearning_DF(X, y, r, numEpochs):
    w = np.zeros(X.shape[1])
    for t in range(numEpochs):
        for i, row in X.iterrows():
            if (np.dot(row, w)*y[i] <= 0):
                w = w + r*y[i]*row
    return w


SP_train.diagnosis.replace(0, -1, inplace = True)
SP_test.diagnosis.replace(0, -1, inplace = True)

#%%
w = perceptronLearning_DF(SP_train.iloc[:,1:22], 
                          SP_train.iloc[:, 0], r = .1, numEpochs=20)
w



#%%
X = np.array([
    [-2,4,-1],
    [4,1,-1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],

])

y = np.array([-1,-1,1,1,1])

def perceptron_nparray(X, Y):
    w = np.zeros(X.shape[1])
    r = 1
    epochs = 20
    for t in range(epochs):
        for i, x in enumerate(X):
            if (np.dot(X[i], w)*Y[i]) < 0:
                w = w + r*X[i]*Y[i]
    return w

w = perceptron_nparray(X,y)
print(w)
