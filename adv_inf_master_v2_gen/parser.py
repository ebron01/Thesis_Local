import nltk
import json
import h5py
import pickle
import numpy as np
# nltk.download('all')
# nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

num_sentences = 10
num_words = 10



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

aux_labels = np.zeros((len(labels_normal_parsed_phrases), len(labels_normal_parsed_phrases[0]), num_sentences, num_words), dtype=np.int32)

for l in range(len(labels_normal_parsed_phrases)):
    for se in range(len(labels_normal_parsed_phrases[l])):
        for s in range(len(labels_normal_parsed_phrases[l][se])):
            if s > 9:
                continue
            aux_labels[l][se][s][:len(labels_normal_parsed_phrases[l][se][s])] = labels_normal_parsed_phrases[l][se][s]

aux_labels_oneword = np.zeros((len(labels_normal_parsed_phrases), len(labels_normal_parsed_phrases[0]), num_sentences, num_words), dtype=np.int32)

for l in range(aux_labels.shape[0]):
    for i in range(aux_labels[l].shape[0]):
        for a in range(aux_labels[l][i].shape[0]):
            if np.count_nonzero(aux_labels[l][i][a]) > 1:
                aux_labels_oneword[l][i][a] = np.zeros((10))
            else:
                aux_labels_oneword[l][i][a] = aux_labels[l][i][a]

with open('/data/shared/ActivityNet/advinf_activitynet/inputs/actnet_gt_np_vp.pkl', 'w') as f:
    pickle.dump(labels_normal_parsed_phrases, f)

with open('/data/shared/ActivityNet/advinf_activitynet/inputs/actnet_gt_np_vp.npy', 'w') as f:
    np.save(f, aux_labels)

with open('/data/shared/ActivityNet/advinf_activitynet/inputs/actnet_gt_np_vp_oneword.npy', 'w') as f:
    np.save(f, aux_labels_oneword)

print('Done')
