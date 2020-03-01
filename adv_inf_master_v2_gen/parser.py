import nltk
import json
import h5py
import pickle

def ie_preprocess(document):
   sentences = nltk.sent_tokenize(document)
   sentences = [nltk.word_tokenize(sent) for sent in sentences]
   sentences = [nltk.pos_tag(sent) for sent in sentences]
   return sentences

input_label_h5 = '/data/shared/ActivityNet/advinf_activitynet/inputs/video_data_dense_label_orj.h5'
print('DataLoader loading h5 file: ', input_label_h5)
h5_label_file = h5py.File(input_label_h5, 'r')
labels = h5_label_file['labels'].value

input_json = '/data/shared/ActivityNet/advinf_activitynet/inputs/video_data_dense_orj.json'
print('DataLoader loading json file: ', input_json)
info = json.load(open(input_json))
ix_to_word = info['ix_to_word']
ix_to_word['1'] = 'raining'
word_to_ix = info['word_to_ix']

labels_word = []
labels_normal = []
for line in labels:
    sentence = []
    normal = []
    for l in line:
        sentence_event = ''
        for w in l:
            if w != 0:
                sentence_event = sentence_event + ' ' + ix_to_word[str(w)]
        sentence_event = str(sentence_event).lstrip(' ')
        sentence.append(sentence_event)
        normal.append(ie_preprocess(sentence_event))
    labels_word.append(sentence)
    labels_normal.append(normal)


'''
<RB.?>* = "0 or more of any tense of adverb," followed by:
<VB.?>* = "0 or more of any tense of verb," followed by:
<NNP>+ = "One or more proper nouns," followed by
<NN>? = "zero or one singular noun."
'''

grammer = """
        NP: {<NN.?>+}
        VP: {<VB.?>*}
        """

cp = nltk.RegexpParser(grammer)

labels_normal_parsed = []
for line in labels_normal:
    parsed = []
    for n in line:
        if n == []:
            parsed.append(n)
            continue
        sentence = []
        result = cp.parse(n[0])
        for a in result:
            if type(a) is nltk.Tree:
                if a.label() == 'NP':  # This climbs into your NVN tree
                    s = ""
                    for i in range(len(a.leaves())):
                        s = s + ' ' + a.leaves()[i][0]
                    sentence.append(s.lstrip())  # This outputs your "NP"
                elif a.label() == 'VP':  # This climbs into your NVN tree
                    s = ""
                    for i in range(len(a.leaves())):
                        s = s + ' ' + a.leaves()[i][0]
                    sentence.append(s.lstrip())  # This outputs your "NP"
        parsed.append(sentence)
    labels_normal_parsed.append(parsed)


labels_normal_parsed_phrases = []
for line in labels_normal_parsed:
    event_phrases = []
    for n in line:
        caption_phrases = []
        for phrases in n:
            p_ = []
            if len(phrases.split()) > 1:
                for p in phrases.split():
                    if p not in ['<', '>', 'unk']:
                        p_.append(word_to_ix[p])
            else:
                if phrases not in ['<', '>', 'unk']:
                    p_.append(word_to_ix[phrases])
            caption_phrases.append(p_)
        event_phrases.append(caption_phrases)
    labels_normal_parsed_phrases.append(event_phrases)

with open('/data/shared/ActivityNet/advinf_activitynet/inputs/actnet_gt_np_vp.pkl', 'w') as f:
    pickle.dump(labels_normal_parsed_phrases, f)

print('Done')

# for line in labels_normal:
#     for n in line:
#         result = cp.parse(n[0][0])
# input_not_parsed_file = 'sorted_10closest_updated_query_mid.json'
# output_parsed_file = 'sorted_10closest_parsed_n_v.json'
#
# # this method tokenizes a document then creates part of speech tags from them and returns pos from documents.
#
#
#
# # this part reads query for middle frame of selected videos.
# with open(input_not_parsed_file, 'r') as f:
#     data = json.load(f)
#
# sentences = []
# sentences_dict = {}
# count = 0
# try:
#     count = 0
#     for key in data.keys():
#         # this part creates a dict of parsed part of speech tags from conceptual caption sentences
#         print(count)
#         if 'concap' in str(key):
#             concap_dict = {}
#             for k in data[key].keys():
#                 # this part encodes sentences with 'utf-8'
#                 # normalized = data[key][k].encode('utf-8')
#                 normalized = data[key][k]['caption']
#                 order = data[key][k]['order']
#                 # sentences.append(ie_preprocess(normalized))
#                 concap_dict.update({k: {'caption': ie_preprocess(normalized), 'order': order}})
#             sentences_dict.update({key: concap_dict})
#         count += 1
# except Exception as e:
#     print(e)

#TODO: Check the grammer rules
# grammer = """
#         NP: {<DT>*<NN.?>*<JJ>*<NN.?>+}
#         VP: {<VB.?>*}
#         """
# parsed_sentences_dict = {}
# for key in sentences_dict.keys():
#     parsed_sentence = {}
#     for k in sentences_dict[key].keys():
#         result = cp.parse(sentences_dict[key][k]['caption'][0])
#         order = sentences_dict[key][k]['order']
#         sentence = []
#         sentence_vp = []
#         for a in result:
#             if type(a) is nltk.Tree:
#                 if a.label() == 'NP':  # This climbs into your NVN tree
#                     s = ""
#                     for i in range(len(a.leaves())):
#                         s = s + ' ' + a.leaves()[i][0]
#                     sentence.append(s.lstrip())  # This outputs your "NP"
#                     # print('np ' + s.lstrip())
#                     # time.sleep(1)
#                 elif a.label() == 'VP':  # This climbs into your NVN tree
#                     s = ""
#                     for i in range(len(a.leaves())):
#                         s = s + ' ' + a.leaves()[i][0]
#                     sentence_vp.append(s.lstrip())  # This outputs your "NP"
#                     # print('vp ' + s.lstrip())
#                     # time.sleep(1)
#         if sentence != []:
#             parsed_sentence.update({k: {'np': sentence, 'vp': sentence_vp, 'order': order}})
#     parsed_sentences_dict.update({key: parsed_sentence})
#
# with open(output_parsed_file, 'w') as f:
#     json.dump(parsed_sentences_dict, f)