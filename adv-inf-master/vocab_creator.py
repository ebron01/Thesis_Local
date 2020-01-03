from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import string

def vocab(activitynet_vocab, concap_vocab):

    actnet_info = json.load(open(activitynet_vocab))
    with open(concap_vocab, 'r') as f:
        concap_info_ = f.readlines()
    actnet_ix_to_word = actnet_info['ix_to_word']
    actnet_word_to_ix = actnet_info['word_to_ix']

    concap_info = []
    for i in range(len(concap_info_)):
        if int(concap_info_[i].split(' ')[1]) > 6: #this lines deletes words that occur less than 6 in Conceptual Captions vocabulary
            concap_info.append(concap_info_[i].split(' ')[0])

    last = 0
    for k in actnet_ix_to_word.keys():
        if int(k) > last:
            last = int(k)
    missing_ix_word = {}
    missing_word_ix = {}
    last = last + 1
    for c_word in concap_info:
        if c_word in string.punctuation:
            continue
        elif c_word == '\'s':
            continue
        if c_word not in actnet_word_to_ix:
            missing_ix_word.update({last: c_word})
            missing_word_ix.update({c_word: last})
            last += 1
    actnet_ix_to_word.update(missing_ix_word)
    actnet_word_to_ix.update(missing_word_ix)
    return actnet_ix_to_word, actnet_word_to_ix


if __name__ == '__main__':

    # activitynet_vocab = 'activity_net/inputs/video_data_dense.json'
    activitynet_vocab = '/data/shared/ActivityNet/activity_net/inputs/video_data_dense.json'
    activitynet_vocab_aux = '/data/shared/ActivityNet/activity_net/inputs/video_data_dense_aux.json'
    concap_vocab = '../glove/vocab.txt'

    actnet_ix_to_word, actnet_word_to_ix = vocab(activitynet_vocab, concap_vocab)
    activitynet_info = {}
    activitynet_info.update({'ix_to_word': actnet_ix_to_word})
    activitynet_info.update({'word_to_ix': actnet_word_to_ix})

    # save full vocab
    with open(activitynet_vocab_aux, 'w') as f:
        json.dump(activitynet_info, f)
