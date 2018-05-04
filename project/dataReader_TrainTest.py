#%%
#  Data Preprocessing In[1 - 11]

# In[1]:
# library preparation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import tensorflow as tf
import re
import time

import random
from ggplot import *


# In[2]:
# data preprocessing In[2-11]

lines = open("movie_lines.txt", encoding="utf-8", errors="ignore").read().split("\n")



convers = open("movie_conversations.txt", encoding="utf-8", errors="ignore").read().split("\n")

#create dict to map id to each line, a correct input to output
id2line = {}
for line in lines:
    _line = line.split(" +++$+++ ")  # a temporary variable
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

#create a list of conversations
converId = []
for conver in convers[:-1]:
    _conver = conver.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
    converId.append(_conver.split(","))


# randomly select a subset of coversations    
converId = random.sample(converId, int(len(converId) * .01))


# questions and anwsers in conversations, note that they are overlapping
questions = []
answers = []
for conver in converId:
    for index in range(len(conver) - 1):
        questions.append(id2line[conver[index]])
        answers.append(id2line[conver[index + 1]])
        

# In[3]:

# clean the texts
#print(answers[0:10])

target = [s for s in questions if "\'" in s]

# define a cleaning text function
def cleanText(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"you ' re", "you are", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"[!@#$%^&*()_+={}|:;\'\"?<>,./~`]", "", text)
    return text

# Create two lists of cleaned Q(question) and A(answer)
QsClean = [cleanText(line) for line in questions]
AsClean = [cleanText(line) for line in answers]



lenQs = [len(Q.split()) for Q in QsClean]
lenAs = [len(A.split()) for A in AsClean]
lenPd = pd.DataFrame({'Q': lenQs, 'A': lenAs})
lenPd.describe()

ggplot(lenPd, aes(x = 'Q')) + geom_histogram(binwidth = 5)
ggplot(lenPd, aes(x = 'A')) + geom_histogram(binwidth = 5)

shortLenPd = lenPd[(lenPd['Q'] <= 15) & (lenPd['Q'] >= 2) \
                    & (lenPd['A'] <= 15) & (lenPd['A'] >= 2)]

ggplot(shortLenPd, aes(x = 'Q')) + geom_histogram(color = 'darkgray', 
      fill = 'white', binwidth = 1) + scale_fill_brewer()
ggplot(shortLenPd, aes(x = 'Q')) + geom_histogram(binwidth = 1)

#plt.hist(shortLenPd['Q'], bins = 14)




# filtering out way too short and long questions and answers

minLength = 2
maxLength = 15

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


h1 = ggplot(pd.DataFrame(wordCounts, columns = ['wC']), aes(x = 'wC')) +\
    geom_histogram()
h2 = ggplot(pd.DataFrame(np.log(wordCounts), columns = ['wC']), aes(x = 'wC')) +\
    geom_histogram(binwidth = .5)
print(h2)

# a: shape para, loc: location para, scale: scale para
gammaA, gammaLoc, gammaScale = ss.gamma.fit(np.log(wordCounts))

myHist = plt.hist(np.log(wordCounts), 20, density = True)
rv = ss.gamma(gammaA, loc = gammaLoc, scale = gammaScale)
x = np.linspace(0.1, 12, 35)
plt.plot(x, rv.pdf(x), lw = 2)
plt.show()


# In[7]:

# define the threshold above,  remove Qs and As with word count less than thQ and thA
thQs = 5
QsWords2Integer = {}
wordInt = 0
for word, count in word2count.items():
    if count >= thQs:
        QsWords2Integer[word] = wordInt
        wordInt += 1

thAs = 5
AsWords2Integer = {}
wordInt = 0
for word, count in word2count.items():
    if count >= thAs:
        AsWords2Integer[word] = wordInt
        wordInt += 1


# add the last tokens to the two dicts above

# PAD:    , EOS: end of string, SOS: start of the string
# OUT: by which all the words that were filtered out by our previous dictionaries will be replaced
tokens = ["<PAD>", "<EOS>", "<OUT>", "<SOS>"]
for token in tokens:
    QsWords2Integer[token] = len(QsWords2Integer) + 1
for token in tokens:
    AsWords2Integer[token] = len(AsWords2Integer) + 1



# In[8]:

## Create the inverse dict of AsWords2Integer dict for decoder part
AsIntegers2Word = {w_i: w for w, w_i in AsWords2Integer.items()}


# In[9]:

# add <EOS> token to the end of every answer sentence
# be careful about the space before the token <EOS>
for i in range(len(AsClean)):
    AsClean[i] += ' <EOS>'


# In[10]:

# translate all the questions and answers into integers that in In[7]
# and replace all the words that were filtered out by <OUT>
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
for length in range(1, 15 + 1):
    for index, Q in enumerate(Qs2Integers):
        if len(Q) == length:
            sortedQsClean.append(Qs2Integers[index])
            sortedAsClean.append(As2Integers[index])


#%%
# Build the seq2seq model In[12 - 17]


# In[12]:


# Create tensorflow placeholders for inputs and targets
# [None, None] is an empty two-dimensional matrix
# None indicates that the first dimension, corresponding to the batch size, can be of any size.
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

    encoderEmbeddedInput = tf.contrib.layers.embed_sequence(inputs,
                                                          AnumWords + 1,
                                                          encoderEmbeddingSize,
                                                          initializer = tf.random_uniform_initializer(0, 1))
    encoderState = encoderRnn(encoderEmbeddedInput, rnnSize, numLayers, kP, seqLength)
    preprocessedTargets = preprocessTargets(targets, QsWords2Integer, batchSize)
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

epochs = 10
batchSize = 32
rnnSize = 256
numLayers = 3
encodingEmbeddingSize = 256
decodingEmbeddingSize = 256

learningRate = 0.01
learningRateDecay = 0.9  #  learning rate dacay over time
minLearningRate = 0.001  # a lower bound for decay

keepProbability = 0.5


# In[19]:

## Create a intercative session And get the training and test predictions

# Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()

# Loading the model inputs,
inputs, targets, lR, kP = modelInputs()

# Setting the sequence length
seqLength = tf.placeholder_with_default(25, None, name = 'seqLength')

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
    optimizer = tf.train.AdamOptimizer(learningRate)
    gradients = optimizer.compute_gradients(lossError)
    # 5. using float
    clippedGradients = [(tf.clip_by_value(gradTensor, -5., 5.), gradVariable)
                        for gradTensor, gradVariable in gradients if gradTensor is not None]
    optimizerGradFlipping = optimizer.apply_gradients(clippedGradients)

# In[21]:

## Padding the sequence with <PAD> token so the all and As and Qs have the same length

# return a seq of intergerID sequences(answers and questions)
def applyPadding(batchOfSeqs, word2Int):
    maxSeqLength = max([len(seq) for seq in batchOfSeqs])
    return [seq + [word2Int["<PAD>"]] * (maxSeqLength - len(seq)) for seq in batchOfSeqs]


# In[22]:

## split the data into batches of Questions and Answers

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

batchIndexCheckTrainingLoss = 5  # Check the training loss every 100 batches

# halfway at the end of one epoch, -1 to round it down
batchIndexCheckValidationLoss = ( (len(trainingQs)) // batchSize // 2 ) - 1

totalTrainingLossError = 0
listValidationLossError = []

# two early stopping variables: check if there is still improvement
earlyStoppingCheck = 0
earlyStoppingStop = 8

# save the weights whenever we want to chat with the chatbot
checkpoint = "chatbotWeights.ckpt"

session.run(tf.global_variables_initializer())

for epoch in range(1, epochs + 1):

    for batchIndex, (paddedQsInbatch, paddedAsInBatch) in enumerate(
            splitIntoBatches(trainingQs, trainingAs, batchSize)):

        startingTime = time.time()
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
                  '''.format(epoch,
                             epochs,
                             batchIndex,
                             len(trainingQs) // batchSize,
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