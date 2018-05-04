#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 20:16:50 2018

@author: LZQ
"""

#%%
import random
import math
import bnNB
#%%
# Define a parse text function and return a 2d list

def parseTxtSPECT(filename):
    dataset = open(filename).read().split("\n")
    for i in range(len(dataset)):
        dataset[i] = dataset[i].split(",")
        for j in range(len(dataset[i])):
            dataset[i][j] = int(dataset[i][j])
    return dataset

Strain = parseTxtSPECT("SPECTtrain.txt")
Stest = parseTxtSPECT("SPECTtest.txt")


#%%
# Initialize the label randomly
# Prob of label 1 
nrow = len(Strain)
ncol = len(Strain[0])

# initial parameter for navie bayesian
q0 = {}
# the keys are just classes
q0['1'] = [0.5 for _ in range(ncol)]
#q0['1'] = [random.random() for _ in range(ncol)]
q0['0'] = [1 - q0['1'][i] for i in range(len(q0['1']))]

T = 100

qTest = {}
qTest['1'] = []
qTest['0'] = []

#%%

noise = 1e-2

for t in range(T):
    
    prob1 = []
    for i in range(nrow): 
        prob1_0 = math.log(q0['1'][0] + noise)
        for j in range(ncol - 1): 
            prob1_0 += math.log(bnNB.Bernoulli(Strain[i][j+1], q0['1'][j + 1]) + noise)
        prob1.append(prob1_0)
        
    prob0 = []
    for i in range(nrow): 
        prob0_0 = math.log(q0['0'][0] + noise)
        for j in range(ncol - 1): 
            prob0_0 += math.log(bnNB.Bernoulli(Strain[i][j+1], q0['0'][j + 1]) + noise)
        prob0.append(prob0_0)
    
    delta = {}
    delta['1'] = [prob1[i]/(prob1[i] + prob0[i]) for i in range(nrow)]
    delta['0'] = [prob0[i]/(prob1[i] + prob0[i]) for i in range(nrow)]
    
    
    q = {}
    q['1'] = []
    q['1'].append(bnNB.mean(delta['1']))
    for j in range(ncol - 1):
        sumdeltaPart = 0
        for i in range(nrow):  
            if (Strain[i][j + 1] == 1):
                sumdeltaPart += delta['1'][i]
        q['1'].append( sumdeltaPart / sum(delta['1']))
        
    q['0'] = []
    q['0'].append(bnNB.mean(delta['0']))
    for j in range(ncol - 1):
        sumdeltaPart = 0
        for i in range(nrow):  
            if (Strain[i][j + 1] == 0):
                sumdeltaPart += delta['0'][i]
        q['0'].append( sumdeltaPart / sum(delta['0'])) 
    
    qTest['1'].append(q['1'])
    qTest['0'].append(q['0'])

    
    error = 0
    for i in range(ncol):
        error += (q['1'][i] - q0['1'][i])**2
    MSE = error**.5 / ncol
    if MSE < 1e-8:
        break
    q0 = q

EM_NBFit = [{}, {}]
EM_NBFit[0][0] = q0['0'][1:]
EM_NBFit[0][1] = q0['1'][1:]

EM_NBFit[1][0] = q0['0'][0]
EM_NBFit[1][1] = q0['1'][0]

#%%

def EM_NB(fit, testset):
    predTest = bnNB.bernoulliNBPredTestset(fit, testset)
    accu = bnNB.accuracyNB(predTest, testset)
    return accu, predTest

#%% 

accu, predTest = EM_NB(EM_NBFit, Stest)
print(accu)

  