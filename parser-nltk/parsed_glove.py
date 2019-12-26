import json
import os
import numpy as np
import datetime
import sys
try:
    import _pickle as pickle
except Exception as e:
    # import pickle
    # print ("Warning: You are running code with Python 2.xx.\n")
    print ("Error: Please run code with Python 3.xx.\n")
    sys.exit(1)
'''
run with CLAS env
'''

dir_path = os.path.dirname(os.path.realpath(__file__))
input_parsed_file = '/sorted_10closest_parsed_np_vp.json'
output_parsed_glove = 'gloves_10closest.pkl'
concurrence_filename = 'concurrence_np_vp_10closest.json'
path_adv_inf_master = '../adv-inf-master/ConCap/inputs/gloves_10closest.pkl'
option = 'pickle' # or 'numpy'

start = datetime.datetime.now()


def loadglovemodel(glovefile):
    print("Loading Glove Model")
    # with open(gloveFile,'r+') as f:
    with open(glovefile, 'r+', encoding='utf-8') as f:
        data = f.readlines()
    model = {}
    try:
        for line in data:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
    except Exception as e:
        print(e)
    print("Done. " + str(len(model)) + " words loaded!\n")
    return model


Model = loadglovemodel(dir_path + "/vectors.txt")


# with open(dir_path + parsed_file, 'r+') as f:
with open(dir_path + input_parsed_file, 'r', encoding='utf-8') as f:
    parsed_sentences_dict = json.load(f)
print('Loaded parsed sentences of closest captions')

# TODO:in progress:
concurrence_d = {}
for key in parsed_sentences_dict.keys():
    np_list = []
    vp_list = []
    for k in parsed_sentences_dict[key].keys():
        for NP in parsed_sentences_dict[key][k]['np']:
            np_list.append(NP)
        for VP in parsed_sentences_dict[key][k]['vp']:
            vp_list.append(VP)
    concurrence_d.update({key: {'np': np_list, 'vp': vp_list}})


concurrence_count = {}
for key in concurrence_d.keys():
    concur_np = {}
    concur_vp = {}
    for _NP in concurrence_d[key]['np']:
        count = 0
        for _np in concurrence_d[key]['np']:
            if str(_NP) == str(_np):
                count += 1
        concur_np.update({_NP: count})
    for _VP in concurrence_d[key]['vp']:
        count = 0
        for _vp in concurrence_d[key]['vp']:
            if str(_VP) == str(_vp):
                count += 1
        concur_vp.update({_VP: count})
    concurrence_count.update({key: {'np': concur_np, 'vp': concur_vp}})

with open(concurrence_filename, 'w') as f:
    json.dump(concurrence_count, f)


# TODO:in progress
for key in parsed_sentences_dict.keys():
    for k in parsed_sentences_dict[key].keys():
        np_glove = []
        vp_glove = []
        for NP in parsed_sentences_dict[key][k]['np']:
            parsed = []
            parsed = NP.split(' ')
            glove = np.zeros(512)
            count = 0
            for i in range(len(parsed)):
                try:
                    glove += Model[parsed[i]]
                    count += 1
                except Exception as e:
                    continue
            glove = glove / count
            np_glove.append(glove)
        for VP in parsed_sentences_dict[key][k]['vp']:
            parsed_v = []
            parsed_v = VP.split(' ')
            glove_v = np.zeros(512)
            count_v = 0
            for i in range(len(parsed_v)):
                try:
                    glove_v += Model[parsed_v[i]]
                    count_v += 1
                except Exception as e:
                    continue
            glove_v = glove_v / count_v
            vp_glove.append(glove_v)
        np_glove = np.asarray(np_glove)
        vp_glove = np.asarray(vp_glove)
        if option == '.npy':
            filename = 'numpys_10/' + str(key) + '_' + str(k)
            np.save(filename + '_np', np_glove)
            np.save(filename + '_vp', vp_glove)
        parsed_sentences_dict[key][k].update({'np_glove': np_glove, 'vp_glove': vp_glove})

end = datetime.datetime.now()
print(end - start)

start = datetime.datetime.now()
if option == 'pickle':
    with open(output_parsed_glove, mode="wb") as opened_file:
        pickle.dump(parsed_sentences_dict, opened_file)
    with open(path_adv_inf_master, mode="wb") as f_opened_file:
        pickle.dump(parsed_sentences_dict, f_opened_file)
    end = datetime.datetime.now()
    print(end - start)

# with open("gloves.pkl", mode="rb") as opened_file:
#     glove_dict1 = pickle.load(opened_file)
# print('Done')
