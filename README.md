# Thesis_Local
# Thesis
1. Enter midframeextractor folder.use midframecreate.py to get mid frames for each event in a given video. you will have 
frame_counts.json, vdata_info.json created and code will gather middle frames for each event into mid folder.

2. Use imagesimilarity.py in turi folder to load conceptual captions saved reference data and save the model for 
comparison as reference_data_all.model.

3. Use modelloader_mid.py to get updated_query_all.json which has distance between comparisons of ConCap dataset and 
ActivityNet dataset event middle frames. 

4. Use parser-nltk/parser.py to get np and vp of closest captions of Concap according to grammer rules defined in parser.py
(pickle_converter.py is written for conversion from pickle3 to pickle2)


5. then use parsed_glove.py to get glove.pickle to use as input in adv-inf.


#TODO: reorder closest captions according to wmd distances.
#TODO2: get best np and vp according to their occurrence counts.


