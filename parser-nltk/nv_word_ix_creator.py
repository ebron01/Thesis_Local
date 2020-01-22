import json
import h5py
import string
filename = 'caption_np_vp_pairs.json'
filename_ = '../adv-inf-master/activity_net/inputs/caption_np_vp_pairs_ix.json'
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


with open(filename_, 'w') as f:
    json.dump(data, f)

print('Done')