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
import pdb
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
import json
start_nb = datetime.now()
download('stopwords')

def removestopwords(captions):
    stop_words = stopwords.words('english')
    normcaptions = {}
    for key in captions.keys():
        line = captions[key].lower().split()
        line = [w for w in line if w not in stop_words]
        normcaptions[key] = line
    return normcaptions

def distance(normcaptions, model):
    distance = np.zeros((len(normcaptions), len(normcaptions)))
    for i in range(len(normcaptions)):
        for j in range(len(normcaptions)):
            distance[i][j] = model.wmdistance(normcaptions[i], normcaptions[j])
    return distance

def _distance(base, captions, model):
    distance = np.zeros((len(captions.keys()), len(captions.keys())))
    key_pairs = {}
    for i, i_key in enumerate(captions.keys()):
        key_pair = {}
        for j, j_key in enumerate(captions.keys()):
            distance[i][j] = model.wmdistance(captions[i_key], captions[j_key])
            key_pair.update({j: j_key})
        key_pairs.update({i: {'key': i_key, 'compared_with': key_pair}})
    return distance, key_pairs

def sortarray(distance,captions):
    summedarray = np.sum(distance, axis=1)
    sortindex = np.argsort(summedarray)
    summedarraysorted = np.array(summedarray)[sortindex.astype(int)]
    sortedarray = np.array(captions)[sortindex.astype(int)]
    return sortedarray

def _sortarray(distance, key_pairs, captions):
    summedarray = np.sum(distance, axis=1)
    sorted = np.sort(summedarray)
    sorted_index = []
    for i in summedarray:
        itemindex = np.where(sorted == i)
        sorted_index.append(itemindex[0][0])
    caption_ordered ={}
    for i in range(len(sorted_index)):
        key = key_pairs[i]['key']
        caption = captions[key]
        caption_ordered.update({key: {'caption': caption, 'order': sorted_index[i]}})
    return caption_ordered

start_embeddings = datetime.now()
print('cell started at : ' + str(start_embeddings))

#this part loads a word2vec model
print('Loading model')
model= KeyedVectors.load_word2vec_format('./inputs/GoogleNews-vectors-negative300.bin', binary=True)
start_embeddings = datetime.now()
print('Loaded model')
print('cell started at : ' + str(start_embeddings))
model.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.
start_embeddings = datetime.now()
print ('normalizing vectors')
print('cell started at : ' + str(start_embeddings))

query = 'updated_query_mid.json'

with open(query, 'r') as f:
    query = json.load(f)

captions_ordered = {}
for key in query.keys():
    base = {}
    if 'concap' not in key:
        base.update({key : query[key]})
        captions = query[key + '_concap']
        captions = removestopwords(captions)
        base = removestopwords(base)
        distance, key_pairs = _distance(base, captions, model)
        caption_ordered = _sortarray(distance, key_pairs, captions)
        captions_ordered.update({key: base, (key + '_concap'): caption_ordered})
    print (key)


with open('sorted_query_mid.json', 'w') as f:
    json.dump(captions_ordered, f)

with open('../parser-nltk/sorted_query_mid.json', 'w') as f:
    json.dump(captions_ordered, f)
# #this part creates the caption array to be reordered after a query on images from Conceptual Captions datasets retrieved images.
# base = ['a big door is being opened in a video game']
# captions = loader("v2.txt")
# #base video caption is added to array t sort
# captions = base + captions[40:50]
# #this removes stopwords from the captions. Because these words do not have any effect on distance.
# normcaptions = removestopwords(captions)
#
# distance = distance(captions, model)
# sortedarray = sortarray(distance,captions)
#
# with open('reordered.txt','a+') as f:
#     for i in range(len(sortedarray)):
#         f.write(sortedarray[i])
#         f.write('\n')
#     f.write('-------------------')
#     f.write('\n')
# print('Cell took %s seconds to run.' % str(datetime.now()- start_embeddings))