#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 21:43:45 2018

@author: LZQ
"""

#%%

##import the data SPECT

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

## Perceptron Learning

SP_train.diagnosis.replace(0, -1, inplace = True)
SP_test.diagnosis.replace(0, -1, inplace = True)
# r is the learning rate and Epochs is just the number of epochs 
def percepLearn_DF(X_train, y_train, X_test, y_test, r, Epochs):
    w = np.zeros(X_train.shape[1])
    for t in range(Epochs):
        for i, row in X_train.iterrows():
#            w = w + r*y_train[i]*row if (np.dot(row, w)*y_train[i] <= 0) else w
            if (np.dot(row, w)*y_train[i] <= 0):
                w = w + r*y_train[i]*row
                
    fit = X_test.dot(w)
    fit[fit<=0] = -1
    fit[fit>0] = 1
    
    E = np.sum(fit == y_test)/len(y_test)
    return [w, E]



#%%

## Implementation of the perceptron learning algorithm 
    
nE = np.array([10, 20, 50, 100, 300, 500])
lR = np.array([.01, .1, 1])

weights = np.zeros((len(nE)*len(lR),22))
accuracyRate = np.zeros(len(nE)*len(lR))

for i in range(len(nE)):
    for j in range(len(lR)):
        weights[j+i*len(lR)], accuracyRate[j+i*len(lR)] = percepLearn_DF(X_train = SP_train.iloc[:,1:23], 
               y_train = SP_train.iloc[:, 0], 
               X_test = SP_test.iloc[:,1:23], y_test = SP_test.iloc[:,0], 
               r = lR[j], Epochs=nE[i]) 
        print("Fitted weights of the perceptron"
              "with leanring rate %.2f and Epochs %d" %(lR[j], nE[i]) + 
              " are:\n {}".format(weights[j+len(lR)*i]) +
              "\n Accuracy Rate is %.3f" %(accuracyRate[j+len(lR)*i]))
        

#%%
#arDF = pd.DataFrame({'learningRate': np.tile(np.array([0.1,1,10]), 3),
#                     'Epochs': np.repeat(np.array([10,20,30]), 3),
#                     'accuracyRate': accuracyRate})
#
#arDF.groupby('Epochs')['accuracyRate'].plot(legend=True)
        
## chart for Comparision of Accuracy Rates on Different Learning Rates and Epochs        
arDF = pd.DataFrame(accuracyRate.reshape((len(nE), len(lR))), 
                    columns=lR,
                    index=nE)
arDF

#%%

## plot for Comparision of Accuracy Rates on Different Learning Rates and Epochs        

#%matplotlib inline
plt = arDF.plot(title='Plot for Comparision of'
                'Accuracy Rates on Different Learning Rates and Epochs')
plt.set_xlabel('epochs')
plt.legend(['learning rate: 0.1', 'learning rate: 1', 'learning rate: 10'])

           


