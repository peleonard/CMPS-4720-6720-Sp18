#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 08:57:21 2018

@author: pli3
"""

#%%
#  Data Preprocessing In[1 - 11]

# In[1]:
# library preparation

import numpy as np
import pandas as pd
import tensorflow as tf
import re
import time
import random


# In[2]:
# data preprocessing In[2-11]

lines = open("movie_lines.txt", encoding="utf-8", errors="ignore").read().split("\n")

convers = open("movie_conversations.txt", encoding="utf-8", errors="ignore").read().split("\n")

#create dict mapping: ids of line to sentence(questions and answers)
id2line = {}
for line in lines:
    _line = line.split(" +++$+++ ")  # a temporary variable
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

#create a list of conversations
converId = []
for conver in convers[:-1]:
    #choose the last one and remove the left and right square bracket
    _conver = conver.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
    converId.append(_conver.split(","))


# # randomly select a subset of coversations   
# random.seed(101)
# converId = random.sample(converId, int(len(converId) * .01))


# questions and anwsers in conversations, note that they are overlapping
Qs = []
As = []
for conver in converId:
    for index in range(len(conver) - 1):
        Qs.append(id2line[conver[index]])
        As.append(id2line[conver[index + 1]])
        

# In[3]:

# clean the texts
#print(answers[0:10])

# pick up potential contraction, say with '
target = [s for s in Qs if "\'s" in s]

# define a cleaning text function
# common short forms 
# http://speakspeak.com/resources/english-grammar-rules/various-grammar-rules/short-forms-contractions
def cleanText(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text

# Create two lists of cleaned Q(question) and A(answer)
QsClean = [cleanText(line) for line in Qs]
AsClean = [cleanText(line) for line in As]

lenQs = [len(Q.split()) for Q in QsClean]
lenAs = [len(A.split()) for A in AsClean]
lenPd = pd.DataFrame({'Q': lenQs, 'A': lenAs})
lenPd.describe()

# from ggplot import *
# ggplot(lenPd, aes(x = 'Q')) + geom_histogram(binwidth = 5)
# ggplot(lenPd, aes(x = 'A')) + geom_histogram(binwidth = 5)

# shortLenPd = lenPd[(lenPd['Q'] <= 15) & (lenPd['Q'] >= 2) \
#                     & (lenPd['A'] <= 15) & (lenPd['A'] >= 2)]

# ggplot(shortLenPd, aes(x = 'Q')) + geom_histogram(color = 'darkgray', 
#       fill = 'white', binwidth = 1) + scale_fill_brewer()
# ggplot(shortLenPd, aes(x = 'Q')) + geom_histogram(binwidth = 1)

#plt.hist(shortLenPd['Q'], bins = 14)




# filtering out way too short and long questions and answers

minLength = 2
maxLength = 25

shortQs = []
shortAs = []


i = 0
for Q in QsClean:
    if minLength <= len(Q.split()) <= maxLength:
        shortQs.append(Q)
        shortAs.append(AsClean[i])
    i += 1

QsClean = []
AsClean = []
i = 0
for A in shortAs:
    if minLength <= len(A.split()) <= maxLength:
        AsClean.append(A)
        QsClean.append(shortQs[i])
    i += 1


# In[4]:

# Create a dictionary mapping word to its count, this is just used in NLP
# which can be used to removed the word under a threshold of frequency
word2count = {}
for Q in QsClean:
    for word in Q.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
for A in AsClean:
    for word in A.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1


# In[5]

## rpy2 library

#import rpy2.robjects as robjects
#robjects.r("pi")
#r = robjects.r

# In[6]:

# Observe the statistical distributions of word counts

# Extract values from a dict and assign it to a list
wordCounts = list(word2count.values())
np.array(wordCounts).shape # .shape is a function in numpy and pandas
len(wordCounts) # len() is a system function

# from ggplot import *
# h1 = ggplot(pd.DataFrame(wordCounts, columns = ['wC']), aes(x = 'wC')) +\
#     geom_histogram()
# h2 = ggplot(pd.DataFrame(np.log(wordCounts), columns = ['wC']), aes(x = 'wC')) +\
#     geom_histogram(binwidth = .5)
# print(h1)
# print(h2)

## a: shape para, loc: location para, scale: scale para
#import matplotlib.pyplot as plt
#import scipy.stats as ss
#gammaA, gammaLoc, gammaScale = ss.gamma.fit(np.log(wordCounts))
#ss.poisson.fit(wordCouts)
#
#myHist = plt.hist(np.log(wordCounts), 20, density = True)
#rv = ss.gamma(gammaA, loc = gammaLoc, scale = gammaScale)
#x = np.linspace(0.1, 12, 35)
#plt.plot(x, rv.pdf(x), lw = 2)
#plt.show()


# In[7]:

pd.DataFrame(wordCounts).describe()

# define the threshold above,  remove Qs and As with word count less than thQ and thA
# And build a dict thats maps each word to a unique integer 
minCountQs = 5
QsWords2Integer = {}
wordInt = 0
for word, count in word2count.items():
    if count >= minCountQs:
        QsWords2Integer[word] = wordInt
        wordInt += 1

minCountAs = 5
AsWords2Integer = {}
wordInt = 0
for word, count in word2count.items():
    if count >= minCountAs:
        AsWords2Integer[word] = wordInt
        wordInt += 1


# add the last tokens to the two dicts above
# PAD: padding token    EOS: end of string     SOS: start of the string
# OUT: by which all the words that were filtered out by our previous dictionaries will be replaced
tokens = ["<PAD>", "<EOS>", "<OUT>", "<SOS>"]
for token in tokens:
    QsWords2Integer[token] = len(QsWords2Integer) + 1
for token in tokens:
    AsWords2Integer[token] = len(AsWords2Integer) + 1


# In[8]:

## Create the inverse dict of AsWords2Integer dict for decoder part
AsIntegers2Word = {wInt: w for w, wInt in AsWords2Integer.items()}


# In[9]:

# add <EOS> token to the end of every answer sentence
# be careful about the space before the token <EOS>
for i in range(len(AsClean)):
    AsClean[i] += ' <EOS>'


# In[10]:

# translate all the questions and answers into integers that in In[7]
# and replace all the words that were filtered out (namely with count < 5) by <OUT>
Qs2Integers = []
for Q in QsClean:
    ints = []
    for word in Q.split():
        if word not in QsWords2Integer:
            ints.append(QsWords2Integer['<OUT>'])
        else:
            ints.append(QsWords2Integer[word])
    Qs2Integers.append(ints)

As2Integers = []
for A in AsClean:
    ints = []
    for word in A.split():
        if word not in AsWords2Integer:
            ints.append(AsWords2Integer['<OUT>'])
        else:
            ints.append(AsWords2Integer[word])
    As2Integers.append(ints)



# In[11]

# Sorts and questions and answers by the length of questions
# It will reduce the amount of padding

sortedQsClean = []
sortedAsClean = []

# restrict the length from 1 to some number (just include not too long questions)
# note that upper bound of range is excluded, thus the maximum length 25
for length in range(minLength, maxLength + 1):
    for index, Q in enumerate(Qs2Integers):
        if len(Q) == length:
            sortedQsClean.append(Qs2Integers[index])
            sortedAsClean.append(As2Integers[index])


#%%
# Build the seq2seq model In[12 - 17]


# In[12]:

# Create 4 tensorflow placeholders that can be fed into the session run
            
# [None, None] is an empty two-dimensional matrix
# None indicates that the size is flexible
#   the first dimension, corresponding to the batch size, can be of any size.
# kP: keep probability parameter
def modelInputs():
    inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    lR = tf.placeholder(tf.float32, name="learningRate")
    kP = tf.placeholder(tf.float32, name="kP")
    return inputs, targets, lR, kP

# In[13]:

# Preprocessing the targets: create the batch,
# and add the SOS to the start of each answer in the batch

# [0, 0] [batchSize, -1] together, 
# means to extract all the lines, except the last word ("<EOS>"). See In[9]
    
def preprocessTargets(targets, word2integer, batchSize):
    leftSide = tf.fill([batchSize, 1], word2integer["<SOS>"])
    rightSide = tf.strided_slice(targets, [0,0], [batchSize, -1], [1, 1])
    preprocessedTargets = tf.concat([leftSide, rightSide], axis=1)
    # axis = 1 means a horizontal concantenation (default is column (axis = 0))
    return preprocessedTargets


# In[14]:

# rnnInputs: everything in the return of modelInputs function, not questions input
# rnnSize: the number of tensors of encoder RNN layer

# kP: keepProbability control the dropout accuracy, It is a dropout regularization in RNN
# seqLength: a list of   length of each question in the batch
def encoderRnn(rnnInputs, rnnSize, numLayers, kP, seqLength):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnnSize)
    lstmDropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=kP)
    encoderCell = tf.contrib.rnn.MultiRNNCell([lstmDropout]*numLayers) # A stacked LSTM
    # not a simple record but a dynamic version of RNN
    encoderOutput, encoderState = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoderCell,
                                                      cell_bw = encoderCell,
                                                      sequence_length = seqLength,
                                                      inputs = rnnInputs,
                                                      dtype = tf.float32)
    return encoderState


# In[15]:

# Decode the RNN layer
  # an attentional decoder function for the training of our dynamic RNN decoder

#1.Decode the training set:

# embedding: "Embeddings" mapping from word to a vec of real numbers
# decoderEmbeddedInput: the Input where we do embedding on  
# decodingScope: "tf.variable_scope"
  
# outputFunction: return the decoder output
  
def decodeTrainingSet(encoderState, decoderCell, decoderEmbeddedInput,
                      seqLength, decodingScope, outputFunction, kP, batchSize):

    attentionStates = tf.zeros([batchSize, 1, decoderCell.output_size])

    (attentionKeys, attentionValues,
    attentionScoreFunctions, attentionConstructFunctions) = tf.contrib.seq2seq.prepare_attention(
            attentionStates, attention_option = 'bahdanau',
            num_units = decoderCell.output_size)

    trainingDecoderFunction = tf.contrib.seq2seq.attention_decoder_fn_train(
            encoderState[0], attentionKeys, attentionValues,
            attentionScoreFunctions, attentionConstructFunctions,
            name = "attn_dec_train")

    decoderOutput, decoderFinalState, decoderFinalContextState = tf.contrib.seq2seq.dynamic_rnn_decoder(
            decoderCell,
            trainingDecoderFunction,
            decoderEmbeddedInput,
            seqLength,
            scope = decodingScope)

    decoderOutputDropout = tf.nn.dropout(decoderOutput, kP)

    return outputFunction(decoderOutputDropout)


#2.Decode the testing/validation set:

# "attention_decoder_fn_train"
def decodeTestSet(encoderState, decoderCell,
                  decoderEmbeddingsMatrix, sosId, eosId, maxLength, numWords,
                  decodingScope, outputFunction, kP, batchSize):

    attentionStates = tf.zeros([batchSize, 1, decoderCell.output_size])

    attentionKeys, attentionValues, attentionScoreFunctions, attentionConstructFunctions = tf.contrib.seq2seq.prepare_attention(
            attentionStates, attention_option = 'bahdanau',
            num_units = decoderCell.output_size)

    testDecoderFunction = tf.contrib.seq2seq.attention_decoder_fn_inference(
            outputFunction,
            encoderState[0], attentionKeys, attentionValues,
            attentionScoreFunctions, attentionConstructFunctions,
            decoderEmbeddingsMatrix, sosId, eosId, maxLength, numWords,
            name = "attn_dec_inf")

    testPredictions, decoderFinalState, decoderFinalContextState = tf.contrib.seq2seq.dynamic_rnn_decoder(
            decoderCell,
            testDecoderFunction,
            scope = decodingScope)

    return testPredictions

# In[16]:

# numWords: num of words in our corpus of answers.
def decoderRnn(decoderEmbeddedInput, decoderEmbeddingsMatrix, encoderState, numWords,
               seqLength, rnnSize, numLayers, word2integer, kP, batchSize):

    with tf.variable_scope("decoding") as decodingScope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnnSize)
        lstmDropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=kP)
        decoderCell = tf.contrib.rnn.MultiRNNCell([lstmDropout]*numLayers)
        
        # weights and biases for fully connected layers
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        outputFunction = lambda x: tf.contrib.layers.fully_connected(x,
                                                                     numWords,
                                                                     None,
                                                                     scope=decodingScope,
                                                                     weights_initializer=weights,
                                                                     biases_initializer=biases)
        
        trainingPredictions = decodeTrainingSet(encoderState,
                                                decoderCell,
                                                decoderEmbeddedInput,
                                                seqLength,
                                                decodingScope,
                                                outputFunction,
                                                kP,
                                                batchSize)
        decodingScope.reuse_variables()
        testPredictions = decodeTestSet(encoderState,
                                        decoderCell,
                                        decoderEmbeddingsMatrix,
                                        word2integer["<SOS>"],
                                        word2integer["<EOS>"],
                                        seqLength - 1,
                                        numWords,
                                        decodingScope,
                                        outputFunction,
                                        kP,
                                        batchSize)
    return trainingPredictions, testPredictions


# In[17]:

# Building the seq2seq model
def seq2seqModel(inputs, targets, kP, batchSize, seqLength, AnumWords, QnumWords,
                 encoderEmbeddingSize, decoderEmbeddingSize, rnnSize, numLayers,
                 QsWords2Integer):
    
    # define a random uniform initializer to train the embeddings also. 
    encoderEmbeddedInput = tf.contrib.layers.embed_sequence(inputs,
                                                          AnumWords + 1,
                                                          encoderEmbeddingSize,
                                                          initializer = tf.random_uniform_initializer(0, 1))
    
    encoderState = encoderRnn(encoderEmbeddedInput, rnnSize, numLayers, kP, seqLength)
    
    preprocessedTargets = preprocessTargets(targets, QsWords2Integer, batchSize)
    
    # We would also train the Embedding Matrix
    decoderEmbeddingsMatrix = tf.Variable(tf.random_uniform([QnumWords + 1, decoderEmbeddingSize], 0, 1))
    
    decoderEmbeddedInput = tf.nn.embedding_lookup(decoderEmbeddingsMatrix, preprocessedTargets)
    
    trainingPredictions, testPredictions = decoderRnn(decoderEmbeddedInput,
                                                      decoderEmbeddingsMatrix,
                                                      encoderState,
                                                      QnumWords,
                                                      seqLength,
                                                      rnnSize,
                                                      numLayers,
                                                      QsWords2Integer,
                                                      kP,
                                                      batchSize)
    
    
    return trainingPredictions, testPredictions



#%%
# training the seq2seq model In[18 - 23]

# In[18]:

# Setting the hyperparameters

epochs = 20
batchSize = 32
rnnSize = 1024
numLayers = 3
encodingEmbeddingSize = 1024
decodingEmbeddingSize = 1024

learningRate = 0.001
learningRateDecay = 0.9  #  learning rate dacay over time
minLearningRate = 0.0001  # a lower bound for decay

keepProbability = 0.5


# In[19]:

## Create a intercative session on which all training will run.

# Defining a tensorflow session
tf.reset_default_graph()
session = tf.InteractiveSession()

# Setting the model inputs by running the 'modelInputs' function
inputs, targets, lR, kP = modelInputs()

# Setting the sequence length, maxLength can be seen as the default value
# tutorial: A placeholder op that passes through input when its output is not fed.
seqLength = tf.placeholder_with_default(maxLength, None, name = 'seqLength')

# Getting the shape of the inputs tensor
inputShape = tf.shape(inputs)

trainingPredictions, testPredictions = seq2seqModel(tf.reverse(inputs, [-1]),
                                                   targets,
                                                   kP,
                                                   batchSize,
                                                   seqLength,
                                                   len(AsWords2Integer),
                                                   len(QsWords2Integer),
                                                   encodingEmbeddingSize,
                                                   decodingEmbeddingSize,
                                                   rnnSize,
                                                   numLayers,
                                                   QsWords2Integer)



# In[20]:

## Loss error, Optimizer(adam optimizer)
## and Gradient Flipping(a trick that will cap the gradient in a range to prevent
## exploding and vanishing issues)

with tf.name_scope("optimization"):
    lossError = tf.contrib.seq2seq.sequence_loss(trainingPredictions,
                                                 targets,
                                                 tf.ones([inputShape[0], seqLength]))
    optimizer = tf.train.AdamOptimizer(lR)
    gradients = optimizer.compute_gradients(lossError)
    # 5. using float
    clippedGradients = [(tf.clip_by_value(gradTensor, -5., 5.), gradVariable)
                        for gradTensor, gradVariable in gradients if gradTensor is not None]
    optimizerGradFlipping = optimizer.apply_gradients(clippedGradients)

# In[21]:

## Padding the sequence with <PAD> token so the all and As or Qs have the same length resp.

# return a seq of intergerID sequences(answers and questions)
def applyPadding(batchOfSeqs, word2Int):
    maxSeqLength = max([len(seq) for seq in batchOfSeqs])
    return [seq + [word2Int["<PAD>"]] * (maxSeqLength - len(seq)) for seq in batchOfSeqs]


# In[22]:

## split the data into batches of Questions and Answers
# finally convert to numpy array in order to continue to use in tensorflow
def splitIntoBatches(Qs, As, batchSize):
    for batchIndex in range(0, len(Qs) // batchSize):
        startIndex = batchIndex * batchSize
        QsInBatch = Qs[startIndex : startIndex + batchSize]
        AsInBatch = As[startIndex : startIndex + batchSize]
        paddedQsInBatch = np.array(applyPadding(QsInBatch, QsWords2Integer))
        paddedAsInBatch = np.array(applyPadding(AsInBatch, AsWords2Integer))
        yield paddedQsInBatch, paddedAsInBatch


## split the Qs and As, into training and validation sets

trainingValidationSplit = int(len(sortedQsClean) * 0.15)
trainingQs = sortedQsClean[trainingValidationSplit:]
trainingAs = sortedAsClean[trainingValidationSplit:]

validationQs = sortedQsClean[:trainingValidationSplit]
validationAs = sortedAsClean[:trainingValidationSplit]


# In[23]:

## Final training

batchIndexCheckTrainingLoss = 100  # Check the training loss every "batchIndexCheckTrainingLoss" batches

# halfway at the end of one epoch, -1 to round it down
batchIndexCheckValidationLoss = ( len(trainingQs) // batchSize // 2 ) - 1

totalTrainingLossError = 0
listValidationLossError = []

# two early stopping variables: check if there is still improvement
earlyStoppingCheck = 0
earlyStoppingStop = 100

# save the weights whenever we want to chat with the chatbot
checkpoint = "chatbotWeights.ckpt"

# initial all global variables
session.run(tf.global_variables_initializer())

for epoch in range(1, epochs + 1):

    for batchIndex, (paddedQsInbatch, paddedAsInBatch) in enumerate(
            splitIntoBatches(trainingQs, trainingAs, batchSize)):

        startingTime = time.time()
        
        # Note "optimizerGradFlipping" and "lossError" are under the optimazation scope 
        # inputs is needed since we have inputShape under this scope, and kP is under the trainingPredictions
        _, batchTrainingLossError = session.run(
                [optimizerGradFlipping, lossError], {inputs: paddedQsInbatch,
                                                     targets: paddedAsInBatch,
                                                     lR: learningRate,
                                                     seqLength: paddedAsInBatch.shape[1],
                                                     kP: keepProbability}) # remeber keepProbability = 0.5
        totalTrainingLossError += batchTrainingLossError
        endingTime = time.time()
        batchTime = endingTime - startingTime


        # average of training loss errors on 100 batches,
        # {:>3} be padded to length 3;
        # {:>6.3f} Combine truncating with padding: six spaces with 3 decimals
        if batchIndex % batchIndexCheckTrainingLoss == 0:
            print('''
                  Epoch: {:>3}/{},
                  Batch: {:>4}/{},
                  Training Loss Error: {:>6.3f},
                  Training Time on 100 Batches: {:d} seconds
                  '''.format(epoch, epochs,
                             batchIndex, len(trainingQs) // batchSize,
                             totalTrainingLossError / batchIndexCheckTrainingLoss,
                             int(batchTime * batchIndexCheckTrainingLoss)))
            totalTrainingLossError = 0  # reinitialize

        # check if it is on halfway
        if batchIndex % batchIndexCheckValidationLoss == 0 and batchIndex > 0:
            totalValidationLossError = 0
            startingTime = time.time()
            for batchIndexValidation, (paddedQsInbatch, paddedAsInBatch) in enumerate(
                    splitIntoBatches(validationQs, validationAs, batchSize)):
                
                # optimizerGradFlipping and
                # kP = 1: kP is only used in training, neouron has always to be present
                batchValidationLossError = session.run(lossError, {inputs: paddedQsInbatch,
                                                                   targets: paddedAsInBatch,
                                                                   lR: learningRate,
                                                                   seqLength: paddedAsInBatch.shape[1],
                                                                   kP: 1})
                totalValidationLossError += batchValidationLossError
            endingTime = time.time()
            batchTime = endingTime - startingTime
            # total validation error / number of batches in this “validation for loop”
            averageValidationLossError = totalValidationLossError / (len(validationQs) / batchSize)
            # {:d} an integer
            print('''Validation Loss Error: {:>6.3f},
                  Batch Validation Time: {:d} seconds'''.format(averageValidationLossError,
                                                                 int(batchTime)))


            learningRate *= learningRateDecay
            if learningRate < minLearningRate:
                learningRate = minLearningRate
                
            listValidationLossError.append(averageValidationLossError)
            if averageValidationLossError <= min(listValidationLossError):
                print('I speak better now!')
                earlyStoppingCheck = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                earlyStoppingCheck += 1
                if earlyStoppingCheck == earlyStoppingStop:
                    break

    if earlyStoppingCheck == earlyStoppingStop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break

print("Game Over")

 #%%

## testing In[24 - 25]



# In[24]:

## Testing

# load the previous weights and run the interractive session
checkpoint = "./chatbotWeights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)

# Convert the questions,  from string to integers
def converString2Int(Q, word2Int):
    question = cleanText(Q)
    
    # get method of dict in Python, to replace the word not in dict with <OUT> token
    return [word2Int.get(word, word2Int['<OUT>']) for word in question.split()]


# In[25]:

# Check the vocabulary of Questions again
#QsWords2Integer
#trainingQs



## Setting up the final chatbot

while(True):
    Q = input("You: ")
    if Q == "Goodbye":
        break
    Q = converString2Int(Q, QsWords2Integer)

    # Padding the question to maxLength by <PAD> token
    Q = Q + [QsWords2Integer["<PAD>"]] * (maxLength - len(Q))

    # neuron network only accept batch of questions
    fakeBatch = np.zeros((batchSize, maxLength))
    fakeBatch[0] = Q
    # testPrediction built it part 2,
    predictedA = session.run(testPredictions, {inputs: fakeBatch, kP: 0.5})[0]

    # postprocess the answer in a clean format
    A = ""
    for i in np.argmax(predictedA, axis = 1):

        if AsIntegers2Word[i] == "i":
            token = " I"
        elif AsIntegers2Word[i] == "<EOS>":
            token = "."
        elif AsIntegers2Word[i] == "<OUT>":
            token = "out"
        else:
            token = " " + AsIntegers2Word[i]
        A += token
        if token == ".":
            break
    print("ChatBot: " + A)