import json
import os
import numpy as np

# vector_file = '../glove/vectors.txt'
# vectors_dict = 'glove_dict.json'
# with open(vector_file, 'r') as f:
#     vectors = f.readlines()
#
# vector_dict = {}
# for v in vectors:
#     v = v.strip()
#     v = v.split(' ')
#     vector_dict.update({v[0]: v[1:]})
#
# with open(vectors_dict, 'w') as f:
#     json.dump(vector_dict, f)

dir_path = os.path.dirname(os.path.realpath(__file__))
parsed_file = 'parsed_np.json'



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

with open(parsed_file, 'w') as f:
    parsed_sentences_dict = json.load(f)


