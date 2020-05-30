#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
************************************USE turi env**********************
Created on Sun Apr 21 21:14:21 2019
@author: emre
Created on thursday 21th of april after thesis meeting.
For wmd sentence comparison wmd can be used. To re-order the captions of retrieved images, wmd will be used over sentences.
This tutorial is implemented to be able to understand and train myself on wmd more.
We can still use the wmd in this file for this task.
Tutorial is in this address:
https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html
@author: emre
"""
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk import download
from pyemd import emd
from gensim.similarities import WmdSimilarity
from datetime import datetime
import pdb
import os
import numpy as np
import json
import logging
import string
from gensim.models import Word2Vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

start_nb = datetime.now()
download('stopwords')

def removestopwords(captions):
    stop_words = stopwords.words('english')
    captions = captions.translate(str.maketrans('','',string.punctuation))
    line = captions.lower().split()
    line = [w for w in line if w not in stop_words]

    return line

start_embeddings = datetime.now()
print('cell started at : ' + str(start_embeddings))

# this part loads a word2vec model
print('Loading model')
# model= KeyedVectors.load_word2vec_format('/media/luchy/HDD/wmd/GoogleNews-vectors-negative300.bin', binary=True)
model = Word2Vec.load("word2vec_custom.model")
start_embeddings = datetime.now()
print('Loaded model')
print('cell started at : ' + str(start_embeddings))
model.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.
start_embeddings = datetime.now()
print ('normalizing vectors')
print('cell started at : ' + str(start_embeddings))

experiment_name = 'concat_mmu'
caption_file_name = 'caption_result_concat_mmu_19Mart_d19.json'

caption_dir = 'adv_inf_master_v2_concat/densevid_eval/'
generated_captions = caption_dir + caption_file_name
gt = '/data/shared/ActivityNet/activitynet_annotations/ActivityNet_val_1.json'
np_save_path = '/home/luchy/Desktop/results/WMD_Scores/'

generated_captions = json.load(open(generated_captions, 'r'))
generated_captions = generated_captions['results']
gt = json.load(open(gt, 'r'))

gt_sentences = {}
for key in gt.keys():
    gt_sentences.update({key: gt[key]['sentences']})

generated_sentences = {}
for key in generated_captions.keys():
    captions = []
    for i in range(len(generated_captions[key])):
        captions.append(generated_captions[key][i]['sentence'])
    generated_sentences.update({key: captions})

#control for consistency of created event captions
for key in generated_captions.keys():
    if key in gt_sentences.keys():
        if len(gt_sentences[key]) != len(generated_sentences[key]):
            generated_sentences[key] = generated_sentences[key][:(len(generated_sentences[key]) // 2)]
            # print (key, len(gt_sentences[key]), len(generated_sentences[key]))

WMD_average_npy = np.zeros(len(generated_captions.keys()), dtype=float)
for k, key in enumerate(generated_captions.keys()):
    if key in gt_sentences.keys():
        distances = 0
        for i in range(len(gt_sentences[key])):
            base = removestopwords(gt_sentences[key][i])
            captions = removestopwords(generated_sentences[key][i])
            if key == 'v_92fD8Cy2zL0':
                captions = ['video', 'cuts', 'sitting', 'getting', 'nails', 'done', 'another', 'woman', 'wearing', 'surgical', 'mask']
            distances += model.wmdistance(base, captions)
        WMD_average_npy[k] = (distances / len(generated_sentences[key]))

np.save(np_save_path + experiment_name + 'custom', WMD_average_npy)
#np.save(np_save_path + experiment_name, WMD_average_npy)

with open('/home/luchy/Desktop/results/WMD_Scores/all_average_scores_custom.txt', 'a') as f:
#with open('/home/luchy/Desktop/results/WMD_Scores/all_average_scores.txt', 'a') as f:
    f.write('WMD Average score for experiment ' + experiment_name + ' is: ' + str(WMD_average_npy.sum() / len(generated_captions.keys())) + ' (' + caption_file_name +')')
    f.write('\n')
print('Done')
