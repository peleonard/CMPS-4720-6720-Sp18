#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:18:22 2018

@author: LZQ
"""


#%%
import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

#%%

# Import the training and testing data
spNames = ['F' + str(i) for i in range(23)]
spNames[0] = 'diagnosis'

SP_train = pd.read_csv("SPECTtrain.txt", names=spNames)
SP_train.shape

SP_test = pd.read_csv("SPECTtest.txt", names=spNames)
SP_test.shape


#%%

# convert the object type from DataFrame to TorchTensor

dtype = torch.FloatTensor
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H1, H2, D_out = 80, SP_train.shape[1]-1, 11, 11, 1

# Convert data from Pandas DataFrame to Numpy Array, 
# and then to Torch Tensor, and Then to Torch Varaible
xTorchTensor = torch.from_numpy(np.array(SP_train.iloc[:, 1:23]))
x = Variable(xTorchTensor.type(dtype), requires_grad=False)
yTorchTensor = torch.from_numpy(np.array(SP_train.iloc[:,0]))
y = Variable(yTorchTensor.type(dtype), requires_grad=False)

#%%

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Variables for its weight and bias.

#One Hidden layer fully connected
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.ReLU(),
    torch.nn.Linear(H1, D_out),
)

# Two Hidden layers fully connected
#model = torch.nn.Sequential(
#    torch.nn.Linear(D_in, H1),
#    torch.nn.Linear(H1, H1),
#    torch.nn.Linear(H2, D_out)
#)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(size_average=True)

learning_rate = 5e-4
epoch = 5000


for t in range(epoch):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Variable of input data to the Module and it produces
    # a Variable of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Variables containing the predicted and true
    # values of y, and the loss function returns a Variable containing the
    # loss.
    loss = loss_fn(y_pred, y)
#    print(t, loss.data[0])

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Variables with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Variable, so
    # we can access its data and gradients like we did before.
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data

print("""Training error of the 1 hidden layer neural network, with learning 
      rate %f and epoch %d, is %.4f""" %(learning_rate,epoch,loss.data))


xTestTorchTensor = torch.from_numpy(np.array(SP_test.iloc[:, 1:23]))
xTest = Variable(xTestTorchTensor.type(dtype), requires_grad=False)

yTestTorchTensor = torch.from_numpy(np.array(SP_test.iloc[:,0]))
yTest = Variable(yTestTorchTensor.type(dtype), requires_grad=False) 
 
yTest_pred = model(xTest)
testError =  loss_fn(yTest_pred, yTest) 
print("""Testing error of the 1 hidden layer neural network, with learning 
      rate %f and epoch %d, is %.4f""" %(learning_rate,epoch,testError))

#%%
threshold = .5
yTest_predNP = yTest_pred.data.numpy()
yTest_predNP[yTest_predNP > threshold] = 1
yTest_predNP[yTest_predNP <= threshold] = 0

yTestNP = yTest.data.numpy()

tn, fp, fn, tp = confusion_matrix(yTestNP, yTest_predNP).ravel()
AR = (tn+tp)/(tn+fp+fn+tp)
Sens = tp/(tp+fn)
Spec = tn/(tn+fp)
print("""The accuracy rate is %.3f, the sensitivity is %.3f, the specificity is 
      %.3f""" %(AR, Sens, Spec))




