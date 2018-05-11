# In[1]:
# library preparation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import tensorflow as tf
import re
import time



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



# filtering out way too short and long questions and answers

shortQs = []
shortAs = []

i = 0
for Q in QsClean:
    if 2 <= len(Q.split()) <= 25:
        shortQs.append(Q)
        shortAs.append(AsClean[i])
    i += 1

QsClean = []
AsClean = []
i = 0
for A in shortAs:
    if 2 <= len(A.split()) <= 25:
        AsClean.append(A)
        QsClean.append(shortQs[i])
    i += 1


# In[4]:

# Create a dictionary mapping word to its count
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


from ggplot import *
h1 = ggplot(pd.DataFrame(wordCounts, columns = ['wC']), aes(x = 'wC')) +\
    geom_histogram()
h2 = ggplot(pd.DataFrame(np.log(wordCounts), columns = ['wC']), aes(x = 'wC')) +\
    geom_histogram()

# a: shape para, loc: location para, scale: scale para
gammaA, gammaLoc, gammaScale = ss.gamma.fit(np.log(wordCounts))

myHist = plt.hist(np.log(wordCounts), 20, density = True)
rv = ss.gamma(gammaA, loc = gammaLoc, scale = gammaScale)
x = np.linspace(0.1, 12, 35)
plt.plot(x, rv.pdf(x), lw = 2)
plt.show()


# In[7]:

# define the threshold above,  remove Qs and As with word count less than thQ and thA
thQs = 20
QsWords2Integer = {}
wordInt = 0
for word, count in word2count.items():
    if count >= thQs:
        QsWords2Integer[word] = wordInt
        wordInt += 1

thAs = 20
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
for length in range(1, 25 + 1):
    for index, Q in enumerate(Qs2Integers):
        if len(Q) == length:
            sortedQsClean.append(Qs2Integers[index])
            sortedAsClean.append(As2Integers[index])