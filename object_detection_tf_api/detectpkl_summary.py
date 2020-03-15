import _pickle as pickle
import os

path = '/data/shared/ConceptualCaptions/fastrcnn_features_36/'
dir_list = os.listdir(path)
save_path = '/data/shared/ConceptualCaptions/downloads_26Oct/detections/'

for file in dir_list:
    with (open(path + file), 'rb') as f:
        data = pickle.load(f)
    del data['detection_features']
    with (open(save_path + file), 'wb') as f:
        pickle.dump(data, f)
    print (str(file) % '%s is saved.')

try_dir = '/home/luchy/Desktop/detections/'
try_list = os.listdir(try_dir)

for file in try_list:
    with open(try_dir + file, 'rb') as f:
        data = pickle.load(f)
    print(file)
