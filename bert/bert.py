import json
#from bert_serving.client import BertClient

filename = '../parser-nltk/sorted_5closest_updated_query_mid.json'

with open(filename, 'rb') as f:
    data = json.load(f)


print ('Done')

#bc = BertClient()
#bc.encode(['First do it', 'then do it right', 'then do it better'])

