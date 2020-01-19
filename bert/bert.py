import json
from bert_serving.client import BertClient
import h5py
import numpy as np
filename = '../parser-nltk/sorted_10closest_updated_query_mid.json'
filename_bert = '../parser-nltk/sorted_10closest_bert.pkl'
with open(filename, 'r') as f:
    data = json.load(f)

bc = BertClient()
count = 0
for key in data.keys():
    if 'concap' not in key:
        continue
    count += 1
    print(count)
    encode_list = []
    for k in data[key].keys():
        encode_list.append(data[key][k]['caption'])
    encoded = bc.encode(encode_list)
    for k,v in enumerate(data[key].keys()):
        data[key][v]['bert'] = encoded[k]

with h5py.File('sample.h5', 'w') as f:
    for key in data.keys():
        if 'concap' not in key:
            continue
        grp = f.create_group(key)
        for k in data[key].keys():
            grp.create_dataset(name=k, shape=data[key][k]['bert'].shape,
                       dtype=np.float32,
                       data=data[key][k]['bert'])
print('Done')

