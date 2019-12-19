#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 23:11:25 2019
https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
@author: emre
"""
#%%This part creates assosiated glove vectors of a word.
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

Model = loadGloveModel(dir_path + "/vectors.txt")

#TODO : words must be sent in a file in this setup. This must be changed and must be generic with a method to parse auxilary characters.
with open(dir_path + "/parsed.txt", 'r') as f:
    words = f.readlines()

for i, word in enumerate(words):
    words[i] = word.rstrip().rstrip(',')
    

glove_vecs = []
glove_vecs_numpy = np.zeros((len(words), 512))

for i, word in enumerate(words):
    glove_vec = []
    glove_vec_numpy = np.zeros(512)
    vec_words = word.split()
    for j in range(len(vec_words)):
        glove_vec_numpy += Model[vec_words[j]]
        #glove_vec.append(Model[vec_words[j]])
    glove_vecs_numpy[i] = (glove_vec_numpy / len(word.split()))

    #glove_vecs.append(glove_vec / len(word.split()))


### this prints vector of innings word: print(Model['inning'])
#print("Boyutları : " + str(Model['inning'].shape))

#%%  https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python

###This part prints the closest words in data based on their glove vectors


import pandas as pd
import csv
import numpy as np
glove_data_file = "vectors.txt"
words = pd.read_table(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

#%%
def vec(w):
  return words.loc[w].as_matrix()
#%%
words_matrix = words.as_matrix()
#%%
def find_closest_word(v):
  diff = words_matrix - v
  delta = np.sum(diff * diff, axis=1)
  i = np.argmin(delta)
  '''
  se np.argpartition. It does not sort the entire array. 
  It only guarantees that the kth element is in sorted position 
  and all smaller elements will be moved before it. 
  Thus the first k elements will be the k-smallest elements.
  '''
  idx = np.argpartition(delta, 50)
#  return words.iloc[idx[4]].name, idx
  return words, idx

vec_word = vec("pitch")

word,idx = find_closest_word(vec_word)

for i in range(0,10):
    print("%s",word.iloc[idx[i]].name )
