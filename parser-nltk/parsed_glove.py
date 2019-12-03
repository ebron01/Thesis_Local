import json
import os
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
parsed_file = '/parsed_np.json'

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

Model = loadGloveModel(dir_path + "/vectors.txt")

with open(dir_path + parsed_file, 'r') as f:
    parsed_sentences_dict = json.load(f)

for key in parsed_sentences_dict.keys():
    for k in parsed_sentences_dict[key].keys():
        np_glove = []
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
        # parsed_sentences_dict[key][k].update({'glove': np_glove})
        np_glove = np.asarray(np_glove)
        np.save('gloves/' + key + '_' + k, np_glove)
print ('Done')
