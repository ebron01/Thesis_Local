import json
import os
import numpy as np
import h5py

num_sentences = 10
num_words = 10
num_words_concat = 32
dir_path = os.path.dirname(os.path.realpath(__file__))
input_parsed_file = '/home/luchy/PycharmProjects/Thesis_Local/parser-nltk/sorted_10closest_parsed_n_v.json'
input_not_parsed_file = '/home/luchy/PycharmProjects/Thesis_Local/parser-nltk/sorted_10closest_updated_query_mid.json'
input_json = '/data/shared/ActivityNet/advinf_activitynet/inputs/video_data_dense_orj.json'
input_label_h5 = '/data/shared/ActivityNet/advinf_activitynet/inputs/video_data_dense_label_orj.h5'



with open(input_parsed_file, 'r') as f:
    parsed_sentences_dict = json.load(f)

with open(input_not_parsed_file, 'r') as f:
    not_parsed_sentences_dict = json.load(f)

print('Loaded parsed sentences of closest captions')
print('DataLoader loading json file: ', input_json)
info = json.load(open(input_json))
ix_to_word = info['ix_to_word']
ix_to_word['1'] = 'raining'
word_to_ix = info['word_to_ix']


# open the hdf5 file containing visual features and captions
print('DataLoader loading h5 file: ', input_label_h5)
h5_label_file = h5py.File(input_label_h5, 'r')
labels = h5_label_file['labels'][()]  # .value
video_id = h5_label_file['video_id'][()]  # .value

vid_id = []
for ix in range(len(video_id)):
    v_ix = video_id[ix]
    vid_id.append(info['videos'][v_ix]['id'])
#
# parsed_keys = []
# for key in parsed_sentences_dict.keys():
#     if key[:13] in vid_id and key[:13] not in parsed_keys:
#         parsed_keys.append(key[:13])

# parsed = {}
# for key in parsed_sentences_dict.keys():
#     for k in parsed_sentences_dict[key].keys():
#        list = []
#        if  parsed_sentences_dict[key][k]['order'] == 0 :
#             list.append(parsed_sentences_dict[key][k]['np'])
#             list.append(parsed_sentences_dict[key][k]['vp'])
#             parsed.update({key : list})
#             break

# not_parsed = {}
# try:
#     for key in not_parsed_sentences_dict.keys():
#         if 'concap' in key:
#             for k in not_parsed_sentences_dict[key].keys():
#                list = []
#                if not_parsed_sentences_dict[key][k]['order'] == 0 :
#                     list.append(not_parsed_sentences_dict[key][k]['caption'])
#                     not_parsed.update({key : list})
#                     break
# except :
#     print('done')
#
# # parsed_cc = {}
# # for v in vid_id:
# #     p_cc = {}
# #     for p in parsed.keys():
# #         if v in p:
# #             # parsed_cc.append(p)
# #             p_cc.update({p.rsplit('_', 3)[1]: parsed[p]})
# #     concat_event = []
# #     for i in sorted(p_cc.keys()):
# #         concat = []
# #         for data in p_cc[i]:
# #             concat += data
# #         concat_event.append(concat)
# #     parsed_cc.update({v: concat_event})
#
# not_parsed_cc = {}
# for v in vid_id:
#     not_p_cc = {}
#     for p in not_parsed.keys():
#         if v in p:
#             # parsed_cc.append(p)
#             not_p_cc.update({p.rsplit('_', 3)[1]: not_parsed[p]})
#     concat_event = []
#     for i in sorted(not_p_cc.keys()):
#         concat_event.append(not_p_cc[i])
#     not_parsed_cc.update({v: concat_event})

# with open('parser_cc.json', 'w') as f:
#     json.dump(parsed_cc, f)

# with open('not_parsed_cc.json', 'w') as f:
#     json.dump(not_parsed_cc, f)

with open('parser_cc.json', 'r') as f:
    parsed_cc = json.load(f)

with open('not_parsed_cc.json', 'r') as f:
    not_parsed_cc = json.load(f)

aux_normal_parsed_phrases = []
for key in vid_id:
    if key in parsed_cc.keys():
        event_phrases = []
        for event in parsed_cc[key]:
            caption_phrases = []
            for n in event:
                p_ = []
                for phrases in n.encode('utf-8').split():
                    if phrases not in ['<', '>', 'unk']:
                        try:
                            p_.append(word_to_ix[str(phrases)])
                        except:
                            p_.append(8472)
                            continue
                caption_phrases.append(p_)
            event_phrases.append(caption_phrases)
        aux_normal_parsed_phrases.append(event_phrases)
    else:
        aux_normal_parsed_phrases.append([])

aux_normal_parsed_concat = []
for key in vid_id:
    if key in parsed_cc.keys():
        event_phrases = []
        for event in parsed_cc[key]:
            caption_phrases = []
            for n in event:
                for phrases in n.encode('utf-8').split():
                    if phrases not in ['<', '>', 'unk']:
                        try:
                            caption_phrases.append(word_to_ix[str(phrases)])
                        except:
                            caption_phrases.append(8472)
                            continue
            event_phrases.append(caption_phrases)
        aux_normal_parsed_concat.append(event_phrases)
    else:
        aux_normal_parsed_concat.append([])

aux_normal_not_parsed_phrases = []
for key in vid_id:
    if key in not_parsed_cc.keys():
        event_phrases = []
        for event in not_parsed_cc[key]:
            p_ = []
            for phrases in event[0].encode('utf-8').split():
                if phrases not in ['<', '>', 'unk']:
                    try:
                        p_.append(word_to_ix[str(phrases)])
                    except:
                        p_.append(8472)
                        continue
            event_phrases.append(p_)
        aux_normal_not_parsed_phrases.append(event_phrases)
    else:
        aux_normal_not_parsed_phrases.append([])

num_words_full_sent = 30
aux_not_parsed = np.zeros((len(aux_normal_not_parsed_phrases), num_sentences, num_words_full_sent), dtype=np.int32)

for l in range(len(aux_normal_not_parsed_phrases)):
    for se in range(len(aux_normal_not_parsed_phrases[l])):
        if se > 9:
            continue
        if len(aux_normal_not_parsed_phrases[l][se]) > 30:
            aux_normal_not_parsed_phrases[l][se] = aux_normal_not_parsed_phrases[l][se][:30]
        aux_not_parsed[l][se][:len(aux_normal_not_parsed_phrases[l][se])] = aux_normal_not_parsed_phrases[l][se]

with open('/data/shared/ActivityNet/advinf_activitynet/inputs/cc_np_vp_full_sent.npy', 'w') as f:
    np.save(f, aux_not_parsed)

aux_labels = np.zeros((len(aux_normal_parsed_phrases), num_sentences, num_sentences, num_words), dtype=np.int32)

for l in range(len(aux_normal_parsed_phrases)):
    for se in range(len(aux_normal_parsed_phrases[l])):
        if se > 9:
            continue
        for s in range(len(aux_normal_parsed_phrases[l][se])):
            if s > 9:
                continue
            aux_labels[l][se][s][:len(aux_normal_parsed_phrases[l][se][s])] = aux_normal_parsed_phrases[l][se][s]

with open('/data/shared/ActivityNet/advinf_activitynet/inputs/cc_np_vp.npy', 'w') as f:
    np.save(f, aux_labels)


aux_concat = np.zeros((len(aux_normal_parsed_concat), num_sentences, num_words_concat), dtype=np.int32)

for l in range(len(aux_normal_parsed_concat)):
    for se in range(len(aux_normal_parsed_concat[l])):
        if se > 9:
            continue
        aux_concat[l][se][:len(aux_normal_parsed_concat[l][se])] = aux_normal_parsed_concat[l][se]

with open('/data/shared/ActivityNet/advinf_activitynet/inputs/cc_np_vp_concat.npy', 'w') as f:
    np.save(f, aux_concat)


# with open('/data/shared/ActivityNet/advinf_activitynet/inputs/cc_np_vp.npy', 'r') as f:
#     aux_labels = np.load(f)

aux_labels_oneword = np.zeros_like(aux_labels)

for l in range(aux_labels.shape[0]):
    for i in range(aux_labels[l].shape[0]):
        for a in range(aux_labels[l][i].shape[0]):
            if np.count_nonzero(aux_labels[l][i][a]) > 1:
                aux_labels_oneword[l][i][a] = np.zeros((10))
            else:
                aux_labels_oneword[l][i][a] = aux_labels[l][i][a]

with open('/data/shared/ActivityNet/advinf_activitynet/inputs/cc_np_vp_oneword.npy', 'w') as f:
    np.save(f, aux_labels_oneword)

print('Done')

