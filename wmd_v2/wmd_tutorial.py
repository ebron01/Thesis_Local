"""
Created on thursday 18th of april after thesis meeting.
For wmd sentence comparison wmd can be used. To re-order the captions of retrieved images, wmd will be used over sentences.
This tutorial is implemented to be able to understand and train myself on wmd more. 
We can still use the wmd in this file for this task.
Tutorial is in this address:
https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html
@author: emre
"""

from datetime import datetime
start_nb = datetime.now()

#%%

# Initialize logging.
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

sentence_obama = 'Hello Emre, greetings from Thailand'
sentence_president = 'Hi emre, you are welcome to the greatest Thailand'
sentence_obama = sentence_obama.lower().split()
sentence_president = sentence_president.lower().split()

#%%

from nltk.corpus import stopwords
from nltk import download

download('stopwords')

#Remove stopwords
stop_words = stopwords.words('english')
sentence_obama = [w for w in sentence_obama if w not in stop_words]
sentence_president = [w for w in sentence_president if w not in stop_words ]

start_embeddings = datetime.now()
print('cell started at : ' + str(start_embeddings))
import os
from gensim.models import KeyedVectors

if not os.path.exists('/Users/emre/Desktop/Tez/09ThesisCode/wmd/GoogleNews-vectors-negative300.bin'):
    raise ValueError('you need to download the google news model')
    
model= KeyedVectors.load_word2vec_format('/Users/emre/Desktop/Tez/09ThesisCode/wmd/GoogleNews-vectors-negative300.bin', binary=True)
model.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.
print('Cell took %s seconds to run.' % str(datetime.now()- start_embeddings))

#%%

from pyemd import emd
from gensim.similarities import WmdSimilarity
distance = model.wmdistance(sentence_obama, sentence_president)
print 'distance = %.4f' % distance

sentence_orange = 'Oranges are my favorite fruit'
sentence_orange = sentence_orange.lower().split()
sentence_orange = [w for w in sentence_orange if w not in stop_words]

distance = model.wmdistance(sentence_obama, sentence_orange)
print 'distance = %.4f' % distance