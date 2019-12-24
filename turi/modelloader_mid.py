import turicreate as tc
import pdb
import json
import csv
tc.config.set_num_gpus(-1)
#pdb.set_trace()


reference_data=tc.load_sframe('conceptualcaptions_0_2.sframe') #conceptualcaptions_0_2.sframe
loaded_model = tc.load_model('reference_data_all.model')
act_data = tc.image_analysis.load_images('./midframeextractor/mid/')
#!!!!don't forget to define k
k=30
query = loaded_model.query(act_data[:], k=30)
data = {}
key_data = {}
count = 0
for i in range(len(query)):
    count += 1
    query[i]['query_label'] = act_data[query[i]['query_label']]['path']
    query[i]['reference_label'] = reference_data[query[i]['reference_label']]['path']
    print(i)
    key_data.update({query[i]['reference_label'].split('/')[6].split('.')[0]: query[i]['distance']})
    if i != 0 and count % k == 0:
        data.update({query[i]['query_label'].split('/')[3].split('.')[0] : key_data})
        key_data = {}
        count = 0
    #data.append(query[i])
   # data.update({query[i]['query_label']: {query[i]['reference_label'] : query[i]['distance']}})
#{'query_label': 33, 'reference_label': 20276, 'distance': 16.396196788098376, 'rank': 30}
print(len(data))
json_name = 'query_mid_closest%d.json'%k
with open(json_name, 'w') as f:
        json.dump(data, f)

with open(json_name, 'r') as f:
    best_diff = json.load(f)
#pdb.set_trace()
#these are splits in ActivityNet detail json to get captions for videos.
split = {'train': 'train', 'val_1': 'val_1', 'val_2': 'val_2'}

vdata = {}
vkeys = []
for key in split.keys():
    filename = '/data/shared/ConceptualCaptions/keras_rmac/data/activitynet_annotations/ActivityNet_%s.json'%key
    with open(filename, 'r') as f:
        data = json.load(f)
        vdata.update({key: data})

#information about ActivityNet dataset
actnet_filename = './midframeextractor/actnet_video_details.json'
with open(actnet_filename, 'r') as f:
    actnet = json.load(f)

#information about Conceptual Caption dataset
#id naming convention for ids starts from 0.jpg so index will be sufficent to get images.
file = '/data/shared/ConceptualCaptions/downloads_26Oct/downloaders/Train_GCC-training.tsv'
f = open(file, 'r')
print ('file read is successful!')
d = csv.reader(f, delimiter="\t")
data = []
for i in d:
    data.append(i)
upd_act = {}
#pdb.set_trace()
for act_key in sorted(best_diff.keys()):
    video_id = str(act_key.rsplit('_', 2)[0])
    event_count = int(act_key.rsplit('_', 2)[
                          1]) - 1  # extract -1 from event count because while middle frames of events are named, event counts starts from 1. Event index for first event in a video starts from 1.
    frame_id = str(act_key.rsplit('_', 2)[2])
    split = actnet[video_id]['split']
    sentence = vdata[split][video_id]['sentences'][event_count]
    upd_concap = {}
    for concat_key in sorted(best_diff[act_key].items(), key=lambda kv: kv[1]):
        concat_sentence = data[int(concat_key[0])][0]
        upd_concap.update({str(concat_key[0]) : concat_sentence})
    upd_act.update({str(act_key) : sentence, str(act_key) + '_concap' : upd_concap})
json_name = '%d_%dclosest_updated_query_mid.json'%(len(best_diff.keys()), k)
with open(json_name, 'w') as f:
    json.dump(upd_act, f)

