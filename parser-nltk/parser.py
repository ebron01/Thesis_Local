import nltk
import json
from unicodedata import normalize
import time
'''
<RB.?>* = "0 or more of any tense of adverb," followed by:
<VB.?>* = "0 or more of any tense of verb," followed by:
<NNP>+ = "One or more proper nouns," followed by
<NN>? = "zero or one singular noun."
'''
input_not_parsed_file = 'sorted_5closest_updated_query_mid.json'
output_parsed_file = 'sorted_5closest_parsed_np_vp.json'

# this method tokenizes a document then creates part of speech tags from them and returns pos from documents.
def ie_preprocess(document):
   sentences = nltk.sent_tokenize(document)
   sentences = [nltk.word_tokenize(sent) for sent in sentences]
   sentences = [nltk.pos_tag(sent) for sent in sentences]
   return sentences


# this part reads query for middle frame of selected videos.
with open(input_not_parsed_file, 'r') as f:
    data = json.load(f)

sentences = []
sentences_dict = {}
count = 0
try:
    count = 0
    for key in data.keys():
        # this part creates a dict of parsed part of speech tags from conceptual caption sentences
        print(count)
        if 'concap' in str(key):
            concap_dict = {}
            for k in data[key].keys():
                # this part encodes sentences with 'utf-8'
                # normalized = data[key][k].encode('utf-8')
                normalized = data[key][k]['caption']
                order = data[key][k]['order']
                # sentences.append(ie_preprocess(normalized))
                concap_dict.update({k: {'caption': ie_preprocess(normalized), 'order': order}})
            sentences_dict.update({key: concap_dict})
        count += 1
except Exception as e:
    print(e)

#TODO: Check the grammer rules
grammer = """
        NP: {<DT>*<NN.?>*<JJ>*<NN.?>+}
        VP: {<VB.?>*}
        """
cp = nltk.RegexpParser(grammer)

parsed_sentences_dict = {}
for key in sentences_dict.keys():
    parsed_sentence = {}
    for k in sentences_dict[key].keys():
        result = cp.parse(sentences_dict[key][k]['caption'][0])
        order = sentences_dict[key][k]['order']
        sentence = []
        sentence_vp = []
        for a in result:
            if type(a) is nltk.Tree:
                if a.label() == 'NP':  # This climbs into your NVN tree
                    s = ""
                    for i in range(len(a.leaves())):
                        s = s + ' ' + a.leaves()[i][0]
                    sentence.append(s.lstrip())  # This outputs your "NP"
                    # print('np ' + s.lstrip())
                    # time.sleep(1)
                elif a.label() == 'VP':  # This climbs into your NVN tree
                    s = ""
                    for i in range(len(a.leaves())):
                        s = s + ' ' + a.leaves()[i][0]
                    sentence_vp.append(s.lstrip())  # This outputs your "NP"
                    # print('vp ' + s.lstrip())
                    # time.sleep(1)
        if sentence != []:
            parsed_sentence.update({k: {'np': sentence, 'vp': sentence_vp, 'order': order}})
    parsed_sentences_dict.update({key: parsed_sentence})

with open(output_parsed_file, 'w') as f:
    json.dump(parsed_sentences_dict, f)

print('Done')
