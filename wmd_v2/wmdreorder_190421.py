#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:14:21 2019

@author: emre
"""

"""
Created on thursday 21th of april after thesis meeting.
For wmd sentence comparison wmd can be used. To re-order the captions of retrieved images, wmd will be used over sentences.
This tutorial is implemented to be able to understand and train myself on wmd more. 
We can still use the wmd in this file for this task.
Tutorial is in this address:
https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html
@author: emre
"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
from nltk.corpus import stopwords
from nltk import download
from pyemd import emd
from gensim.similarities import WmdSimilarity
import os
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk import download
from datetime import datetime
import numpy as np
start_nb = datetime.now()
download('stopwords')

#this part must be generic. Right now it is according to v2.txt file. May or may not be used depending on v2.txt file.
#TODO: rewrite loader method after creating a pipeline.
def loader(textfile):
    with open(textfile, 'r') as f:
        captions = f.readlines()
    indices = 0, 11,12,23,24,35,36,47,48,59
    captions = [i for j, i in enumerate(captions) if j not in indices]
    for i in range(len(captions)):
        captions[i] = captions[i].split(" ", 1)[1].lstrip().rstrip()
    return captions

def removestopwords(captions):
    stop_words = stopwords.words('english')
    normcaptions = []
    for line in captions:
        line = line.lower().split()
        line = [w for w in line if w not in stop_words]
        normcaptions.append(line)
    return normcaptions

def distance(normcaptions, model):
    distance = np.zeros((len(normcaptions), len(normcaptions)))
    for i in range(len(normcaptions)):
        for j in range(len(normcaptions)):
            distance[i][j] = model.wmdistance(normcaptions[i], normcaptions[j])
    return distance

def sortarray(distance,captions):
    summedarray = np.sum(distance, axis=1)
    sortindex = np.argsort(summedarray)
    summedarraysorted = np.array(summedarray)[sortindex.astype(int)]
    sortedarray = np.array(captions)[sortindex.astype(int)]
    return sortedarray

start_embeddings = datetime.now()
print('cell started at : ' + str(start_embeddings))

if not os.path.exists('/Users/emre/Desktop/Tez/09ThesisCode/wmd/GoogleNews-vectors-negative300.bin'):
    raise ValueError('you need to download the google news model')

#this part loads a word2vec model    
model= KeyedVectors.load_word2vec_format('/Users/emre/Desktop/Tez/09ThesisCode/wmd/GoogleNews-vectors-negative300.bin', binary=True)
model.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.

#this part creates the caption array to be reordered after a query on images from Conceptual Captions datasets retrieved images.
base = ['a big door is being opened in a video game']
captions = loader("v2.txt")
#base video caption is added to array t sort
captions = base + captions[40:50]
#this removes stopwords from the captions. Because these words do not have any effect on distance.
normcaptions = removestopwords(captions)

distance = distance(normcaptions,model)
sortedarray = sortarray(distance,captions) 

with open('reordered.txt','a+') as f:
    for i in range(len(sortedarray)):
        f.write(sortedarray[i])
        f.write('\n')
    f.write('-------------------')
    f.write('\n')
print('Cell took %s seconds to run.' % str(datetime.now()- start_embeddings))


        
