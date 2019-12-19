import os
import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import json
import math
import operator
import datetime
import pdb
print(datetime.datetime.now())
#get information about whole ActivityNet Captions dataset(train,val_1,val_2) included
with open('/data/shared/ConceptualCaptions/keras_rmac/data/activitynet_annotations/video_data_dense.json', 'r') as f:
    video_data = json.load(f)
print('video_data_dense.json loaded')
#get video data details from json
video_data = video_data['videos']
# extracted frames of 10 sample videos from activitynet dataset just to have a good comparison
# filenames_local = os.listdir('/Users/emre/Desktop/Tez/09ThesisCode/GitThesisCode/keras_rmac/ActVideos/frames/')
filenames_server = os.listdir('/data/shared/ActivityNet/Crawler/done_videos')
#with open('filenames_server.txt', 'r') as f:
#    filenames_server = f.readlines()
filenames = filenames_server
print(datetime.datetime.now())
#get 10 video details
vdata = {}
for fn in filenames:
    for vd in video_data:
        if fn.strip() == str(vd['id']):
            vdata.update({vd['id']: vd})
print('vdata info created')
with open('vdata_info.json', 'w') as f:
    json.dump(vdata, f)
print('vdata saved to vdata_info.json')
#this part counts frames in a dir of activitynet videos for each. This count will be used to decide which frame corresponds as middle frame of an event
file_count = {}
frame_path_local = '/Users/emre/Desktop/Tez/09ThesisCode/GitThesisCode/keras_rmac/ActVideos/frames/'
frame_path_server = '/data/shared/ActivityNet/Crawler/done_videos/'
frame_path = frame_path_server

#must uncomment this part for a new frame counts json file
#for fn in filenames:
#    file_count.update({fn: len([name for name in os.listdir(frame_path + fn) if os.path.isfile(frame_path + fn + '/' + name)])})
#with open('frame_counts.json', 'w') as f:
#    json.dump(file_count, f)
#print('frame_counts.json created')

#must comment out this part for a new frame counts json file
with open('frame_counts.json', 'r') as f:
    file_count = json.load(f)
print('frame_counts.json loaded')

print(datetime.datetime.now())
#TODO: there are missing videos when video_data and filenames are compared, find them
#this part adds information about start, middle and end frame of each event in a video of activitynet dataset
for fn in vdata.keys():
    frames = file_count[fn]
    duration = vdata[fn]['duration']
    timestamps = vdata[fn]['timestamps']
    frame_locs = []
    for stamp in timestamps:
        start = stamp[0]
        end = stamp[1]
        start_frame = int(math.floor(start * (frames / duration)))
        start_frame = max(0, start_frame)
        end_frame = int(math.floor(end * (frames / duration)))
        end_frame = min(end_frame, frames)
        middle_frame = int(math.floor((start_frame + end_frame) / 2))
        frame_locs.append([start_frame, middle_frame, end_frame])
    vdata[fn].update({str('frame_locs'): frame_locs})
print('frame locs are added')
print(datetime.datetime.now())
#dump activitynet new information data to use while extracting middle frames' rmac vectors for each event to compare with Conceptual Captions
actnet_filename = '/data/shared/ConceptualCaptions/keras_rmac/data/activitynet_annotations/actnet_video_details.json'
with open(actnet_filename, 'w') as f:
    json.dump(vdata, f)
print('actnet_video_details.json created')
mid_fr_path = '~/Thesis_Local/turi/midframeextractor/mid/'
image_path = '/data/shared/ActivityNet/Crawler/done_videos/'
print(datetime.datetime.now())
for vd in vdata.keys():
        #this frame locs gives info for all sample videos' start middle end frame info.
    frame_locs = vdata[vd]['frame_locs']

        #this part creates new directories for each video to store resulting rmac features for each events' middle frame per sample videos.
    #current_dir = os.getcwd()
    #os.chdir(mid_fr_path)
    #os.system('mkdir %s' % vd)
    #os.chdir(current_dir)

    event_count = 0
        #with this for loop each events middle frame is send to rmac feature extraction.
    for frame in frame_locs:
        event_count += 1
        path = image_path + vd + '/'
        mid_frame_name = 'frame%04d.jpg' % frame[1]
        mid_frame_path = path + mid_frame_name
        outfile = vd + '_' + str(event_count) + '_' + mid_frame_name
        mid_f_r_path = mid_fr_path + '/' + outfile
        #mid_f_r_path = mid_fr_path + vd + '/' + outfile
        os.system('cp %s %s' % (mid_frame_path, mid_f_r_path))
        #os.system('cd %s' % (current_dir + '/mid/' + vd))
        #os.system('find . -empty -type d -delete')
        #os.system('cd %s' % current_dir)
	#outfile = vd + '_' + str(event_count) + '_' + mid_frame_name
        #os.system('mv %s %s' % (mid_f_r_path,  outfile))
