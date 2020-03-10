# import json
# import h5py
#
# # file_path = '/home/luchy/PycharmProjects/Thesis_Local/adv_inf_master_v2_gen/vis/vis_val_result_gen_embed_1_cc_npvp_vectors.json'
# file_path = '/home/luchy/Desktop/results/result_base/vis/vis_val_base_k100.json'
# with open(file_path, 'r') as f:
#     data = json.load(f)
#
# train_json = '/home/luchy/PycharmProjects/Thesis_Local/adv_inf_master_v2_gen/densevid_eval/data/train.json'
# with open(train_json, 'r') as f:
#     gt_train = json.load(f)
#
# input_not_parsed_file = '../../parser-nltk/sorted_10closest_updated_query_mid.json'
# with open(input_not_parsed_file, 'r') as f:
#     cc_notparsed_file = json.load(f)
#
# input_parsed_file = '/home/luchy/PycharmProjects/Thesis_Local/parser-nltk/sorted_10closest_parsed_n_v.json'
# with open(input_parsed_file, 'r') as f:
#     parsed_sentences_dict = json.load(f)
#
# not_parsed = {}
# try:
#     for key in cc_notparsed_file.keys():
#         for k in cc_notparsed_file[key].keys():
#            list = []
#            if  cc_notparsed_file[key][k]['order'] == 0 :
#                 list.append(cc_notparsed_file[key][k]['caption'])
#                 not_parsed.update({key : list})
#                 break
# except:
#     print('Done')
#
# parsed = {}
# for key in parsed_sentences_dict.keys():
#     for k in parsed_sentences_dict[key].keys():
#        list = []
#        if  parsed_sentences_dict[key][k]['order'] == 0 :
#             list.append(parsed_sentences_dict[key][k]['np'])
#             list.append(parsed_sentences_dict[key][k]['vp'])
#             parsed.update({key : list})
#             break
#
# parsed_cc = {}
# for v in vid_id:
#     p_cc = {}
#     for p in parsed.keys():
#         if v in p:
#             # parsed_cc.append(p)
#             p_cc.update({p.rsplit('_', 3)[1]: parsed[p]})
#     concat_event = []
#     for i in sorted(p_cc.keys()):
#         concat = []
#         for data in p_cc[i]:
#             concat += data
#         concat_event.append(concat)
#     parsed_cc.update({v: concat_event})
#
#
# for i in range(2,5):
#      paragraph = paragraph + data[i]['caption'] + '.'
# print(paragraph)
#
# for i in range(2,5):
#      paragraph = paragraph + data[i]['gt'] + '.'
# print(paragraph)
#
#
#
#
# input_label_h5 = '/data/shared/ActivityNet/advinf_activitynet/inputs/video_data_dense_label_orj.h5'
#
#
# h5_label_file = h5py.File(input_label_h5, 'r')
# labels = h5_label_file['labels'][()]
# video_id = h5_label_file['video_id'][()]
#
# print('Done')
#
# info = json.load(open(input_json))
# for i in range(19810):
#     if data[0]['video_id'] == info['videos'][i]['id']:
#         print(i)
#         break
#
# vid_id = info['videos'][v_ix]['id']
#
import json
# file_path = '/home/luchy/PycharmProjects/Thesis_Local/adv_inf_master_v2_gen/vis/vis_val_result_gen_embed_1_cc_npvp_vectors.json'
val_gen_json = '/home/luchy/Desktop/results/result_base/vis/vis_val_base_k100.json'
with open(val_gen_json, 'r') as f:
    val_gen_data = json.load(f)

train_gt_json = '/home/luchy/PycharmProjects/Thesis_Local/adv_inf_master_v2_gen/densevid_eval/data/train.json'
with open(train_gt_json, 'r') as f:
    train_gt_data = json.load(f)

cc_notparsed_json = '../../parser-nltk/sorted_10closest_updated_query_mid.json'
with open(cc_notparsed_json, 'r') as f:
    cc_notparsed_data = json.load(f)

cc_parsed_json = '../../parser-nltk/sorted_10closest_parsed_n_v.json'
with open(cc_parsed_json, 'r') as f:
    cc_parsed_data = json.load(f)

cc_not_parsed_paragraph = {}
for train_key in train_gt_data.keys():
    cc_not_parsed = {}
    for notparsed_key in cc_notparsed_data.keys():
        if train_key in notparsed_key and 'concap' in notparsed_key:
            for cc_closest in cc_notparsed_data[notparsed_key].keys():
                if cc_notparsed_data[notparsed_key][cc_closest]['order'] == 0 :
                    cc_not_parsed.update({notparsed_key.rsplit('_', 3)[1]: cc_notparsed_data[notparsed_key][cc_closest]['caption']})
                    break
    cc_not_parsed_events = []
    for i in sorted(cc_not_parsed.keys()):
        cc_not_parsed_events.append(cc_not_parsed[i])
    cc_not_parsed_paragraph.update({train_key: cc_not_parsed_events})

cc_not_parsed_paragraph_json = 'cc_not_parsed_paragraph_closest1.json'
with open(cc_not_parsed_paragraph_json, 'w') as f:
    json.dump(cc_not_parsed_paragraph, f)

print('Done')