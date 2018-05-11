#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 21:15:42 2018

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
SP_train.diagnosis.replace(0, -1, inplace = True)
SP_test.diagnosis.replace(0, -1, inplace = True)

def perceptronLearning_DF(X, y, r, numEpochs):
    w = np.zeros(X.shape[1])
    for t in range(numEpochs):
        for i, row in X.iterrows():
            if (np.dot(row, w)*y[i] <= 0):
                w = w + r*y[i]*row
    return w
#%%
w = perceptronLearning_DF(SP_train.iloc[:,1:23], 
                          SP_train.iloc[:, 0], r = .1, numEpochs=200)
print(w)
#accuracy rate
def perceptronLearning_AR(X, y, w):
    fit = X.dot(w)
    fit[fit<=0] = -1
    fit[fit>0] = 1
    
    E = np.sum(fit == y)/len(y)
    return E

accuracyRate = perceptronLearning_AR(SP_test.iloc[:,1:23], SP_test.iloc[:,0], w)
accuracyRate    


