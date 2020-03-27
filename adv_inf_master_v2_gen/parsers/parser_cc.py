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
input_cooccur = '/home/luchy/PycharmProjects/Thesis_Local/parser-nltk/concurrence_np_vp_10closest.json'
input_cooccur_new = '/home/luchy/PycharmProjects/Thesis_Local/parser-nltk/concurrence_np_vp_10closest_26Mar.json'

with open(input_parsed_file, 'r') as f:
    parsed_sentences_dict = json.load(f)

with open(input_not_parsed_file, 'r') as f:
    not_parsed_sentences_dict = json.load(f)

with open(input_cooccur, 'r') as f:
    cooccur = json.load(f)


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

# aux_normal_parsed_phrases = []
# for key in vid_id:
#     if key in parsed_cc.keys():
#         event_phrases = []
#         for event in parsed_cc[key]:
#             caption_phrases = []
#             for n in event:
#                 p_ = []
#                 for phrases in n.encode('utf-8').split():
#                     if phrases not in ['<', '>', 'unk']:
#                         try:
#                             p_.append(word_to_ix[str(phrases)])
#                         except:
#                             p_.append(8472)
#                             continue
#                 caption_phrases.append(p_)
#             event_phrases.append(caption_phrases)
#         aux_normal_parsed_phrases.append(event_phrases)
#     else:
#         aux_normal_parsed_phrases.append([])
#
# aux_normal_parsed_concat = []
# for key in vid_id:
#     if key in parsed_cc.keys():
#         event_phrases = []
#         for event in parsed_cc[key]:
#             caption_phrases = []
#             for n in event:
#                 for phrases in n.encode('utf-8').split():
#                     if phrases not in ['<', '>', 'unk']:
#                         try:
#                             caption_phrases.append(word_to_ix[str(phrases)])
#                         except:
#                             caption_phrases.append(8472)
#                             continue
#             event_phrases.append(caption_phrases)
#         aux_normal_parsed_concat.append(event_phrases)
#     else:
#         aux_normal_parsed_concat.append([])
#
# aux_normal_not_parsed_phrases = []
# for key in vid_id:
#     if key in not_parsed_cc.keys():
#         event_phrases = []
#         for event in not_parsed_cc[key]:
#             p_ = []
#             for phrases in event[0].encode('utf-8').split():
#                 if phrases not in ['<', '>', 'unk']:
#                     try:
#                         p_.append(word_to_ix[str(phrases)])
#                     except:
#                         p_.append(8472)
#                         continue
#             event_phrases.append(p_)
#         aux_normal_not_parsed_phrases.append(event_phrases)
#     else:
#         aux_normal_not_parsed_phrases.append([])
#
# num_words_full_sent = 30
# aux_not_parsed = np.zeros((len(aux_normal_not_parsed_phrases), num_sentences, num_words_full_sent), dtype=np.int32)
#
# for l in range(len(aux_normal_not_parsed_phrases)):
#     for se in range(len(aux_normal_not_parsed_phrases[l])):
#         if se > 9:
#             continue
#         if len(aux_normal_not_parsed_phrases[l][se]) > 30:
#             aux_normal_not_parsed_phrases[l][se] = aux_normal_not_parsed_phrases[l][se][:30]
#         aux_not_parsed[l][se][:len(aux_normal_not_parsed_phrases[l][se])] = aux_normal_not_parsed_phrases[l][se]
#
# with open('/data/shared/ActivityNet/advinf_activitynet/inputs/cc_np_vp_full_sent.npy', 'w') as f:
#     np.save(f, aux_not_parsed)
#
# aux_labels = np.zeros((len(aux_normal_parsed_phrases), num_sentences, num_sentences, num_words), dtype=np.int32)
#
# for l in range(len(aux_normal_parsed_phrases)):
#     for se in range(len(aux_normal_parsed_phrases[l])):
#         if se > 9:
#             continue
#         for s in range(len(aux_normal_parsed_phrases[l][se])):
#             if s > 9:
#                 continue
#             aux_labels[l][se][s][:len(aux_normal_parsed_phrases[l][se][s])] = aux_normal_parsed_phrases[l][se][s]
#
# with open('/data/shared/ActivityNet/advinf_activitynet/inputs/cc_np_vp.npy', 'w') as f:
#     np.save(f, aux_labels)
#
#
# aux_concat = np.zeros((len(aux_normal_parsed_concat), num_sentences, num_words_concat), dtype=np.int32)
#
# for l in range(len(aux_normal_parsed_concat)):
#     for se in range(len(aux_normal_parsed_concat[l])):
#         if se > 9:
#             continue
#         aux_concat[l][se][:len(aux_normal_parsed_concat[l][se])] = aux_normal_parsed_concat[l][se]
#
# with open('/data/shared/ActivityNet/advinf_activitynet/inputs/cc_np_vp_concat.npy', 'w') as f:
#     np.save(f, aux_concat)
#
#
# # with open('/data/shared/ActivityNet/advinf_activitynet/inputs/cc_np_vp.npy', 'r') as f:
# #     aux_labels = np.load(f)
#
# aux_labels_oneword = np.zeros_like(aux_labels)
#
# for l in range(aux_labels.shape[0]):
#     for i in range(aux_labels[l].shape[0]):
#         for a in range(aux_labels[l][i].shape[0]):
#             if np.count_nonzero(aux_labels[l][i][a]) > 1:
#                 aux_labels_oneword[l][i][a] = np.zeros((10))
#             else:
#                 aux_labels_oneword[l][i][a] = aux_labels[l][i][a]
#
# with open('/data/shared/ActivityNet/advinf_activitynet/inputs/cc_np_vp_oneword.npy', 'w') as f:
#     np.save(f, aux_labels_oneword)

print('Done')

###PART for COOCCURANCE
# concurrence_d = {}
# for key in parsed_sentences_dict.keys():
#     np_list = []
#     vp_list = []
#     for k in parsed_sentences_dict[key].keys():
#         for NP in parsed_sentences_dict[key][k]['np']:
#             np_list.append(NP)
#         for VP in parsed_sentences_dict[key][k]['vp']:
#             vp_list.append(VP)
#     concurrence_d.update({key: {'np': np_list, 'vp': vp_list}})
#
#
# concurrence_count = {}
#
# for key in concurrence_d.keys():
#     concur_np = {}
#     concur_vp = {}
#     for _NP in concurrence_d[key]['np']:
#         count = 0
#         for _np in concurrence_d[key]['np']:
#             try:
#                 if str(_NP) == str(_np):
#                     count += 1
#             except:
#                 continue
#         concur_np.update({_NP: count})
#     for _VP in concurrence_d[key]['vp']:
#         count = 0
#         for _vp in concurrence_d[key]['vp']:
#             try:
#                 if str(_VP) == str(_vp):
#                     count += 1
#             except:
#                 continue
#         concur_vp.update({_VP: count})
#     concurrence_count.update({key: {'np': concur_np, 'vp': concur_vp}})

# with open(concurrence_filename, 'w') as f:
#     json.dump(concurrence_count, f)

with open(input_cooccur_new, 'r') as f:
    concurrence_count = json.load(f)
#
# count = 0
# cooccur_parsed_cc = {}
# for vid in vid_id:
#     count += 1
#     print(count)
#     p_cc = {}
#     p_cc_keys = []
#     for p in concurrence_count.keys():
#         if vid in p:
#             p_cc.update({int(p.rsplit('_', 3)[1]): concurrence_count[p]})
#             p_cc_keys.append(p)
#     concat_event = []
#     for i in sorted(p_cc.keys()):
#         concat = []
#         for k in p_cc[i].keys():
#             for a, v in sorted(p_cc[i][k].items(), key=lambda item: item[1], reverse=True):
#                 if v > 1:
#                     concat.append(a)
#         for num in range(10):
#             if len(concat) >= 15:
#                 concat = concat[:15]
#                 break
#             for a in parsed_sentences_dict[sorted(p_cc_keys)[int(i)-1]].keys():
#                 if len(concat) >= 15:
#                     concat = concat[:15]
#                     break
#                 if parsed_sentences_dict[sorted(p_cc_keys)[int(i)-1]][a]['order'] == num:
#                     if len(parsed_sentences_dict[sorted(p_cc_keys)[int(i)-1]][a]['np']) > len(
#                             parsed_sentences_dict[sorted(p_cc_keys)[int(i)-1]][a]['vp']):
#                         leng = len(parsed_sentences_dict[sorted(p_cc_keys)[int(i)-1]][a]['np'])
#                     else:
#                         leng = len(parsed_sentences_dict[sorted(p_cc_keys)[int(i)-1]][a]['vp'])
#
#                     for l in range(leng):
#                         try:
#                             if parsed_sentences_dict[sorted(p_cc_keys)[int(i)-1]][a]['np'][l] not in concat:
#                                 concat.append(parsed_sentences_dict[sorted(p_cc_keys)[int(i)-1]][a]['np'][l])
#                         except IndexError:
#                             "Do nothing"
#                         try:
#                             if parsed_sentences_dict[sorted(p_cc_keys)[int(i)-1]][a]['vp'][l] not in concat:
#                                 concat.append(parsed_sentences_dict[sorted(p_cc_keys)[int(i)-1]][a]['vp'][l])
#                         except IndexError:
#                             "Do nothing"
#                         if len(concat) >= 15:
#                             concat = concat[:15]
#                             break
#         concat_event.append(concat)
#     cooccur_parsed_cc.update({vid: concat_event})
#
# for key in cooccur_parsed_cc.keys():
#     for i in range(len(cooccur_parsed_cc[key])):
#         if len(cooccur_parsed_cc[key][i]) < 15:
#             if i == 0:
#                 cooccur_parsed_cc[key][i] = cooccur_parsed_cc[key][i+1]
#             else:
#                 cooccur_parsed_cc[key][i] = cooccur_parsed_cc[key][0]
#
# with open('cc_np_vp_cooccur_words.json', 'w') as f:
#     json.dump(cooccur_parsed_cc, f)


with open('cc_np_vp_cooccur_words.json', 'r') as f:
    cooccur_parsed_cc = json.load(f)

# #This creates a input with parsed with over two words and unk words
# cou = 0
# num_words_cooccur = 15
# cooccur_parsed_cc_phrases = []
# for key in vid_id:
#     if key in cooccur_parsed_cc.keys():
#         event_phrases = []
#         for event in cooccur_parsed_cc[key]:
#             p_ = []
#             for r in range(num_words_cooccur):
#                 if event[r] == u'\uf0a7' or event[r] == u'\xa3' or event[r] == u'\xe9':
#                     # print (event[r])
#                     cou += 1
#                     event[r] = 'babalanga'
#                 if len(event[r].encode('utf-8').split()) > 1:
#                     sum = 0
#                     for q in range(len(event[r].encode('utf-8').split())):
#                         if event[r].encode('utf-8').split()[q] not in ['<', '>', 'unk']:
#                             try:
#                                 sum += (word_to_ix[str(event[r].encode('utf-8').split()[q])] * (10 ** (q * 4)))
#                             except:
#                                 sum += (8472 * (10 ** (q * 4)))
#                     p_.append(sum)
#                 else:
#                     try:
#                         if event[r].encode('utf-8') not in ['<', '>', 'unk']:
#                             try:
#                                 p_.append(word_to_ix[str(event[r].encode('utf-8'))])
#                             except:
#                                 p_.append(8472)
#                                 continue
#                     except:
#                         print(event[r])
#             event_phrases.append(p_)
#         cooccur_parsed_cc_phrases.append(event_phrases)
#     else:
#         cooccur_parsed_cc_phrases.append([])


#This creates a input with parsed with max two words and unk words
cou = 0
num_words_cooccur = 15
cooccur_parsed_cc_phrases = []
for key in vid_id:
    if key in cooccur_parsed_cc.keys():
        event_phrases = []
        for event in cooccur_parsed_cc[key]:
            p_ = []
            for r in range(num_words_cooccur):
                if event[r] == u'\uf0a7' or event[r] == u'\xa3' or event[r] == u'\xe9':
                    # print (event[r])
                    cou += 1
                    event[r] = 'babalanga'

                concatted =''
                if type(event[r]) == list:
                    for i in range(len(event[r])):
                        concatted += event[r][i] + ' '
                    event[r] = concatted

                if len(event[r].split()) > 1:
                    try:
                        event[r] = event[r].encode('utf-8').split()[:2]
                    except:
                        print(event[r])
                        try:
                            event[r] = event[r].split()[0].encode('utf-8').split()[:2]
                            print(event[r])
                        except:
                            event[r] = event[r].split()[1].encode('utf-8').split()[:2]
                            print(event[r])
                    sum = 0
                    for q in range(len(event[r])):
                        if event[r][q] not in ['<', '>', 'unk']:
                            try:
                                sum += (word_to_ix[str(event[r][q])] * (10 ** (q * 4)))
                            except:
                                sum += (8472 * (10 ** (q * 4)))
                    p_.append(sum)
                else:
                    try:
                        if event[r].encode('utf-8') not in ['<', '>', 'unk']:
                            try:
                                p_.append(word_to_ix[str(event[r].encode('utf-8'))])
                            except:
                                p_.append(8472)
                                continue
                    except:
                        print(event[r])
            event_phrases.append(p_)
        cooccur_parsed_cc_phrases.append(event_phrases)
    else:
        cooccur_parsed_cc_phrases.append([])

num_words_cooccur = 15
cooccur_parsed = np.zeros((len(cooccur_parsed_cc_phrases), num_sentences, num_words_cooccur), dtype=np.int32)

try:
    for l in range(len(cooccur_parsed_cc_phrases)):
        for se in range(len(cooccur_parsed_cc_phrases[l])):
            if se > 9:
                continue
            if len(cooccur_parsed_cc_phrases[l][se]) > 15:
                cooccur_parsed_cc[l][se] = cooccur_parsed_cc_phrases[l][se][:15]
            cooccur_parsed[l][se][:len(cooccur_parsed_cc_phrases[l][se])] = cooccur_parsed_cc_phrases[l][se]
except:
    print('Done')
with open('/data/shared/ActivityNet/advinf_activitynet/inputs/cc_np_vp_cooccur.npy', 'w') as f:
    np.save(f, cooccur_parsed)
