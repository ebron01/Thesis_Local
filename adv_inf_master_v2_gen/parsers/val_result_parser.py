import json
import time

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

cc_parsed_closest1_json = 'parser_cc.json'
with open(cc_parsed_closest1_json, 'r') as f:
    cc_parsed_closest1 = json.load(f)

# this part creates paragraphs for each video from conceptual captions dataset for training ids of activitynet dataset
# cc_not_parsed_paragraph = {}
# for train_key in train_gt_data.keys():
#     cc_not_parsed = {}
#     for notparsed_key in cc_notparsed_data.keys():
#         if train_key in notparsed_key and 'concap' in notparsed_key:
#             for cc_closest in cc_notparsed_data[notparsed_key].keys():
#                 if cc_notparsed_data[notparsed_key][cc_closest]['order'] == 0 :
#                     cc_not_parsed.update({notparsed_key.rsplit('_', 3)[1]: cc_notparsed_data[notparsed_key][cc_closest]['caption']})
#                     break
#     cc_not_parsed_events = []
#     for i in sorted(cc_not_parsed.keys()):
#         cc_not_parsed_events.append(cc_not_parsed[i])
#     cc_not_parsed_paragraph.update({train_key: cc_not_parsed_events})

# cc_not_parsed_paragraph_json = 'cc_not_parsed_paragraph_closest1.json'
# with open(cc_not_parsed_paragraph_json, 'w') as f:
#     json.dump(cc_not_parsed_paragraph, f)

cc_not_parsed_paragraph_json = 'cc_not_parsed_paragraph_closest1.json'
with open(cc_not_parsed_paragraph_json, 'r') as f:
    cc_not_parsed_paragraph = json.load(f)
val_gen_concat_mmu_d19 = '/home/luchy/PycharmProjects/Thesis_Local/adv_inf_master_v2_concat/densevid_eval/caption_result_concat_mmu_19Mart_d19.json'
with open(val_gen_concat_mmu_d19, 'r') as f:
    val_gen_concat_mmu = json.load(f)

val_gen_json_missing = '/home/luchy/PycharmProjects/Thesis_Local/adv_inf_master_v2_concat/densevid_eval/caption_result_base_missing_21Mart.json'
with open(val_gen_json_missing, 'r') as f:
    val_gen_data_missing = json.load(f)

val_sent = ''
val_sent_missing = ''
val_gen_concat_mmu_sent = ''
val_id = 'v_5SNtTQZnN4g'

print(
    '----------------------------------------------------------------------------------------------------------------------')
print(val_id)
print(
    '----------------------------------------------------------------------------------------------------------------------')
print('ground truth:')
print(train_gt_data[val_id])
print('\n')

for i in range(len(val_gen_concat_mmu['results'][val_id])):
    val_gen_concat_mmu_sent += val_gen_concat_mmu['results'][val_id][i]['sentence'] + '.'
print('CC Concat MMU prediction:')
print (val_gen_concat_mmu_sent)
print('\n')

for i in range(len(val_gen_data['results'][val_id])):
    val_sent += val_gen_data['results'][val_id][i]['sentence'] + '.'
print('CC Concat prediction:')
print(val_sent)
print('\n')

for i in range(len(val_gen_data_missing['results'][val_id])):
    val_sent_missing += val_gen_data_missing['results'][val_id][i]['sentence'] + '.'
print('Base Model (Missing) Prediction:')
print (val_sent_missing)
print('\n')

print('not parsed')
print (cc_not_parsed_paragraph[val_id])
print('\n')

print('parsed')
print (cc_parsed_closest1[val_id])
print('\n')
time.sleep(3)

print('Done')