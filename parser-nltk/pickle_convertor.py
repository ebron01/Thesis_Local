import _pickle as pickle
import csv
#
data = pickle.load(open('gloves_10closest.pkl', 'rb'))
#
with open("gloves_10closest_v2.pkl", mode="wb") as opened_file:
     pickle.dump(data, opened_file, 2)

#import os
#os.chdir('../adv-inf/save/')

#data = pickle.load(open('infos.pkl', 'rb'), encoding='unicode_escape')

#print ('Done')



