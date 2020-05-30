# import gensim, logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# from nltk.corpus import stopwords
# import string
# import json
# from gensim.models import Word2Vec
#
#
# def removestopwords(captions):
#     stop_words = stopwords.words('english')
#     captions = captions.translate(str.maketrans('','',string.punctuation))
#     line = captions.lower().split()
#     line = [w for w in line if w not in stop_words]
#     return line
#
# caption_file_name = 'caption_result_concat_full_sent_23Mart.json'
#
# caption_dir = 'adv_inf_master_v2_concat/densevid_eval/'
# generated_captions = caption_dir + caption_file_name
# gt = '/data/shared/ActivityNet/activitynet_annotations/ActivityNet_val_1.json'
# np_save_path = '/home/luchy/Desktop/results/WMD_Scores/'
#
# generated_captions = json.load(open(generated_captions, 'r'))
# generated_captions = generated_captions['results']
# gt = json.load(open(gt, 'r'))
#
# gt_sentences = {}
# for key in gt.keys():
#     gt_sentences.update({key: gt[key]['sentences']})
#
# generated_sentences = {}
# for key in generated_captions.keys():
#     captions = []
#     for i in range(len(generated_captions[key])):
#         captions.append(generated_captions[key][i]['sentence'])
#     generated_sentences.update({key: captions})
#
# #control for consistency of created event captions
# for key in generated_captions.keys():
#     if key in gt_sentences.keys():
#         if len(gt_sentences[key]) != len(generated_sentences[key]):
#             generated_sentences[key] = generated_sentences[key][:(len(generated_sentences[key]) // 2)]
# whole = []
# for k, key in enumerate(generated_captions.keys()):
#     if key in gt_sentences.keys():
#         for i in range(len(gt_sentences[key])):
#             base = removestopwords(gt_sentences[key][i])
#             whole.append(base)
# model = Word2Vec(whole, size=100, min_count=1, workers=4)
# model.save("word2vec_custom.model")
# model.save("/home/luchy/Desktop/word2vec_custom.model")
# print('Done')
#
# # sentences = [['first', 'sentence'], ['second', 'sentence']]
# # # train word2vec on the two sentences
# # model = gensim.models.Word2Vec(sentences, min_count=1)
# # vector = model.wv['sentence']
#
# # dict = {}
# # for key in generated_sentences.keys():
# #     for i in range(len(generated_sentences[key])):
# #         captions = generated_sentences[key][i].translate(str.maketrans('', '', string.punctuation))
# #         line = captions.lower().split()
# #         for j in range(len(line)):
# #             if line[j] not in dict.keys():
# #                 dict.update({line[j]:1})
# #             else:
# #                 dict[line[j]] = dict[line[j]] + 1
# #
# # sort_orders = sorted(dict.items(), key=lambda x: x[1], reverse=True)
# #
# # count = 0
# # for key in sort_orders.keys():
# #     if sort_orders[key] > 1:
# #         count += 1
# #
#
# info = json.load(open('/data/shared/ActivityNet/advinf_activitynet/inputs/video_data_dense_orj.json'))
# ix_to_word = info['ix_to_word']
# ix_to_word['1'] = 'raining'
# word_to_ix = info['word_to_ix']
#
#
#
#
import json
from collections import defaultdict
from gensim import corpora

gt = '/data/shared/ActivityNet/activitynet_annotations/ActivityNet_val_1.json'
gt_captions = json.load(open(gt))
text_corpus = []
for key in gt_captions.keys():
    for i in range(len(gt_captions[key]['sentences'])):
        text_corpus.append(gt_captions[key]['sentences'][i])

# Create a set of frequent words
stoplist = set('for a of the and to in'.split(' '))
# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in text_corpus]

# Count word frequencies
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
print(processed_corpus)

dictionary = corpora.Dictionary(processed_corpus)
print(len(dictionary))

print('Done')