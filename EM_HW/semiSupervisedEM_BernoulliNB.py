#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 22:19:26 2018

@author: pli3
"""



import bnNB
import random
import math


#%%
Strain = bnNB.parseTxtSPECT("SPECTtrain.txt")
Stest = bnNB.parseTxtSPECT("SPECTtest.txt")



StrainLabeled = []
for i in range(len(Strain)):
    if i < 4 or i > 75:
        StrainLabeled.append(Strain[i])
 
StrainUnlabeled = Strain[4:76]

semiNBFit0 = bnNB.meanByClass(StrainLabeled)

noise = 1e-2

def llLabeled(fit, datasetLabeled):
    ll = []
    for i in range(len(datasetLabeled)):
        ll0 = fit[1][datasetLabeled[i][0]]
        for j in range(len(datasetLabeled[0]) - 1):
            ll0 *= bnNB.Bernoulli(datasetLabeled[i][j + 1], 
                             fit[0][datasetLabeled[i][j + 1]][j])
        ll.append(math.log(ll0 + 1e-2))
    return(sum(ll))
 
llLabeled(semiNBFit0, StrainLabeled)


def llUnlabeled(fit, probs, datasetUnlabeled):
    ll = []
    for i in range(len(datasetUnlabeled)):
        ll0_0 = probs[i][0]
        ll1_0 = probs[i][0]
        for j in range(len(datasetUnlabeled[0]) - 1):
            ll0_0 *= bnNB.Bernoulli(datasetUnlabeled[i][j + 1],
                               fit[0][0][j])
            ll1_0 *= bnNB.Bernoulli(datasetUnlabeled[i][j + 1],
                               fit[0][1][j])
        
        ll.append(math.log(ll0_0 + ll1_0 + 1e-2))
        
    return(sum(ll))



T = 100
LL = []
error = 1e-10
for t in range(T):
    
    unlabeledProbs = []
    for i in range(len(StrainUnlabeled)):
        unlabeledProb = bnNB.testProb(semiNBFit0, StrainUnlabeled[i][1:])    
        unlabeledProbs.append(unlabeledProb)
    
    unlabeledPred = bnNB.bernoulliNBPredTestset(semiNBFit0, StrainUnlabeled)
     
    for i in range(len(StrainUnlabeled)):
        StrainUnlabeled[i][0] = unlabeledPred[i]
    
    semiNBFit = bnNB.meanByClass(Strain)
    
    loglikeLabeled = llLabeled(semiNBFit, StrainLabeled) 
    loglikeUnlabeled = llUnlabeled(semiNBFit, unlabeledProbs, StrainUnlabeled)
    LL.append(loglikeLabeled + loglikeUnlabeled)
    
    if (t > 0 and ((LL[t] - LL[t-1]) < error)):
        break
    
    semiNBFit0 = semiNBFit



#%%


def semiEM_NB(fit, testset):
    predTest = bnNB.bernoulliNBPredTestset(fit, testset)
    accu = bnNB.accuracyNB(predTest, testset)
    return accu, predTest

accu, predTest = semiEM_NB(semiNBFit, Stest)
print("""The predictions of test dataset and accuracy for a semi-supervised 
      Bernoulli Naive Bayes on SPECT dastet is \n [%s] \n and %.3f""" %(predTest, accu))







