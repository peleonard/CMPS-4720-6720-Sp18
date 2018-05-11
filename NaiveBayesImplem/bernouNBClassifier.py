#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 02:17:00 2018

@author: LZQ
"""

import math

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

# the first column is the response

def sepByClass(dataset):
	dictMatch = {}
	for i in range(len(dataset)):
		row = dataset[i]
		if (row[0] not in dictMatch):
			dictMatch[row[0]] = []
		dictMatch[row[0]].append(row[:-1])
	return dictMatch

def priorProb(trainset):
    trainSep = sepByClass(trainset)
    probClass = {}
    for key in trainSep.keys():
        probClass[key] = float(len(trainSep[key])/len(trainset))
    return probClass


trainSep = sepByClass(Strain)
priorProb(Strain)

#%%

# conditional mean

def mean(numbers):
	return float(sum(numbers)/len(numbers))

def meanCol(trainset):
    means = [mean(attribute) for attribute in zip(*trainset)]
    return means
    
def meanByClass(trainset):
    trainSep = sepByClass(trainset)
    priorProbs = priorProb(trainset)
    meanPerFeature = {}
    for y, features in trainSep.items():
        meanPerFeature[y] = meanCol(features)
    return meanPerFeature, priorProbs

bernoulliNBFit = meanByClass(Strain)


#%%

# conditional prob is bernoulli
def bernoulli(x, p):
    return p**x * (1-p)**(1-x)


# Calculate the posterior probs 
def testProb(NBFit, testFeatures):
    priorProbs = NBFit[1]
    probs = {}
    for y, meanFeature in NBFit[0].items():
        probs[y] = priorProbs[y]
        for i in range(len(meanFeature)):
            probs[y] *= bernoulli(testFeatures[i], meanFeature[i])
        probs[y] = math.log(probs[y] + 1)
    return probs


# Use the probs to predict the response of testset
def bernoulliNBPred(NBFit, testFeatures):
    probs = testProb(NBFit, testFeatures)
    # probPred: final predicted prob
    classPred, probPred = None, -1
    for y, prob in probs.items():
        if classPred is None or prob > probPred:
            probPred = prob
            classPred = y
    return classPred

def bernoulliNBPredTestset(NBFit, testset):
    classPreds = []
    for i in range(len(testset)):
        testFeatures = testset[i][1:]
        classPreds.append(bernoulliNBPred(NBFit, testFeatures))
    return classPreds


def accuracyNB(predTest, testset):
    correctPred = 0
    for i in range(len(testset)):
        if predTest[i] == testset[i][0]:
            correctPred += 1
    return correctPred/len(predTest)


#%%

## A final bernoulli naive bayesian classifier for SPECT dataset
def bernoulliNBClassifier(trainset, testset):
    bernoulliNBFit = meanByClass(trainset)
    predTest = bernoulliNBPredTestset(bernoulliNBFit, testset)
    accu = accuracyNB(predTest, testset)
    return predTest,accu

predictionsTest, accuracy = bernoulliNBClassifier(Strain, Stest)

