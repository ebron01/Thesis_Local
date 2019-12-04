import json
import os
import numpy as np
import _pickle as pickle
import datetime

dir_path = os.path.dirname(os.path.realpath(__file__))
parsed_file = '/parsed_np_vp.json'
option = 'pickle' # or 'numpy'
start = datetime.datetime.now()
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    with open(gloveFile,'r+', encoding='utf-8') as f:
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
    print("Done.",len(model)," words loaded!")
    return model

Model = loadGloveModel(dir_path + "/vectors.txt")

with open(dir_path + parsed_file, 'r', encoding='utf-8') as f:
    parsed_sentences_dict = json.load(f)

glove_dict = {}
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
        # parsed_sentences_dict[key][k].update({'glove': np_glove})
        np_glove = np.asarray(np_glove)
        vp_glove = np.asarray(vp_glove)
        if option == 'numpy':
            np.save('gloves/' + key + '_' + k + '_np', np_glove)
            np.save('gloves/' + key + '_' + k + '_vp', vp_glove)
        elif option == 'pickle':
            glove_dict.update({key + '_' + k + '_vp': vp_glove, key + '_' + k + '_np': np_glove})
end = datetime.datetime.now()
print(end - start)

start = datetime.datetime.now()
if option == 'pickle':
    with open("gloves.pkl", mode="wb") as opened_file:
        pickle.dump(glove_dict, opened_file)
    end = datetime.datetime.now()
    print(end - start)

#with open("gloves.pkl", mode="rb") as opened_file:
#    glove_dict1 = pickle.load(opened_file)
#print('Done')
