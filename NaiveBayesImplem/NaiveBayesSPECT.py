#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 10:53:48 2018

@author: LZQ
"""

#%%
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


#%%

# Import the training and testing data
spNames = ['F' + str(i) for i in range(23)]
spNames[0] = 'diagnosis'

SP_train = pd.read_csv("SPECTtrain.txt", names=spNames)
SP_train.shape

SP_test = pd.read_csv("SPECTtest.txt", names=spNames)
SP_test.shape

#%%

#clf = GaussianNB()
#clf.fit(SP_train.iloc[:, 1:23], SP_train.iloc[:, 0])
#GaussianNB(priors=None)

clf = BernoulliNB()
clf.fit(SP_train.iloc[:, 1:23], SP_train.iloc[:, 0])
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)

#clf = MultinomialNB()
#clf.fit(SP_train.iloc[:, 1:23], SP_train.iloc[:, 0])
#MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)


yTest_predNB = clf.predict(SP_test.iloc[:, 1:23])
print(clf.predict(SP_test.iloc[:, 1:23]))

tn, fp, fn, tp = confusion_matrix(SP_test.iloc[:, 0], yTest_predNB).ravel()
AR = (tn+tp)/(tn+fp+fn+tp)
Sens = tp/(tp+fn)
Spec = tn/(tn+fp)
print("""The accuracy rate is %.3f, the sensitivity is %.3f, the specificity is 
      %.3f""" %(AR, Sens, Spec))


#%%

clf1 = BernoulliNB()
clf1.partial_fit(SP_train.iloc[:, 1:23], SP_train.iloc[:, 0], np.array([0, 1]))
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)

yTest_predNB1 = clf.predict(SP_test.iloc[:, 1:23])
print(clf1.predict(SP_test.iloc[:, 1:23]))

tn, fp, fn, tp = confusion_matrix(SP_test.iloc[:, 0], yTest_predNB1).ravel()
AR = (tn+tp)/(tn+fp+fn+tp)
Sens = tp/(tp+fn)
Spec = tn/(tn+fp)
print("""The accuracy rate is %.3f, the sensitivity is %.3f, the specificity is 
      %.3f""" %(AR, Sens, Spec))