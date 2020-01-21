import json
import h5py
filename = 'concurrence_n_v_10closest.json'
label_file = '../adv-inf-master/activity_net/inputs/video_data_dense.h5'

with open(filename, 'r') as f:
    data = json.load(f)


label = h5py.File(label_file, 'r')

print('Done')