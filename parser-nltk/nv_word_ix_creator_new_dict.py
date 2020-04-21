import json
import h5py
import string


filename = 'caption_np_vp_pairs.json'
parsed_cc = '../adv_inf_master_v2_gen/parsers/parser_cc.json'
filename_ = '/data/shared/ActivityNet/advinf_activitynet/inputs/caption_np_vp_pairs_ix.json'
filename_order = '/data/shared/ActivityNet/advinf_activitynet/inputs/caption_np_vp_pairs_ix_order.json'
label_file = '/data/shared/ActivityNet/advinf_activitynet/inputs/video_data_dense.h5'
video_info = '/data/shared/ActivityNet/advinf_activitynet/inputs/video_data_dense.json'


with open(video_info, 'r') as f:
    video_data = json.load(f)

word_to_ix = video_data['word_to_ix']

# with open(filename, 'r') as f:
#     data = json.load(f)

with open(parsed_cc, 'r') as f:
    parsed_data = json.load(f)

count = 8473
key_num = 0
parsed_data_ix = {}
for key in parsed_data.keys():
    if key_num % 20 == 0:
        print(key_num)
    sent_ = []
    for sent in parsed_data[key]:
        se_ = []
        for se in sent:
            s_ = ''
            for s in se.translate(str.maketrans('', '', string.punctuation)).lower().split():
                if s.strip() in word_to_ix.keys():
                    ix = word_to_ix[s.strip()]
                    s_ = s_ + ' ' + str(ix)
                else:
                    word_to_ix[s.strip()] = count
                    count += 1
                    s_ = s_ + ' ' + str(count)
            se_.append(s_.lstrip())
        sent_.append(se_)
    parsed_data_ix.update({key: sent_})
    key_num += 1


print(count)
with open(filename_, 'w') as f:
    json.dump(parsed_data_ix, f)

with open('new_dict.json', 'w') as f:
    json.dump(word_to_ix, f)

print('Done')
