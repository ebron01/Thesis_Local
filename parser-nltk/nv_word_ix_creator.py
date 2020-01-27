import json
import h5py
import string
filename = 'caption_np_vp_pairs.json'
filename_ = '../adv-inf-master/activity_net/inputs/caption_np_vp_pairs_ix.json'
filename_order = '../adv-inf-master/activity_net/inputs/caption_np_vp_pairs_ix_order.json'
label_file = '../adv-inf-master/activity_net/inputs/video_data_dense.h5'
video_info = '../adv-inf-master/activity_net/inputs/video_data_dense.json'


with open(video_info, 'r') as f:
    video_data = json.load(f)

word_to_ix = video_data['word_to_ix']

with open(filename, 'r') as f:
    data = json.load(f)

for key in data.keys():
    caption = data[key]['caption']
    caption_ix = []
    for c in caption.translate(str.maketrans('', '', string.punctuation)).lower().split():
        if c.strip() in word_to_ix.keys():
            ix = word_to_ix[c.strip()]
            caption_ix.append(ix)
        else:
            caption_ix.append(8472)
    data[key]['caption_ix'] = caption_ix

data_ = {}
for key in data.keys():
    for k in data[key]['_concap']:
        if data[key]['_concap'][k]['order'] == 0:
            caption = data[key]['_concap'][k]['caption']
            caption_ix = []
            for c in caption.translate(str.maketrans('', '', string.punctuation)).lower().split():
                if c.strip() in word_to_ix.keys():
                    ix = word_to_ix[c.strip()]
                    caption_ix.append(ix)
                else:
                    caption_ix.append(8472)
            data_.update({key.rsplit('_', 1)[0]: caption_ix})

data_.update({'v_kMsWDe0V1Xg_3': data_['v_kMsWDe0V1Xg_2']})
with open(filename_, 'w') as f:
    json.dump(data, f)

with open(filename_order, 'w') as f:
    json.dump(data_, f)

print('Done')
