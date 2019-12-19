import turicreate as tc
tc.config.set_num_gpus(-1)
import pdb
import os
# Load images from the downloaded data
#path = '/data/shared/ConceptualCaptions/downloads_26Oct/'
#img_folders = os.listdir(path)
#img_folders.remove('downloaders')
#reference_data_all = []
#for i in range(len(img_folders)):
#    ref_name = 'reference_data' + str(i)
#    print(ref_name)
#    ref_name = tc.image_analysis.load_images(path + str(img_folders[i]))
#    reference_data_all.append(ref_name)
#    print(str(img_folders[i]) + ' appended')

#reference_data_all = reference_data_all.add_row_number()
#reference_data_all.save('./conceptualcaptions_all.sframe')

#reference_data0  = tc.image_analysis.load_images('/data/shared/ConceptualCaptions/downloads_26Oct/images0_100')
#reference_data1  = tc.image_analysis.load_images('/data/shared/ConceptualCaptions/downloads_26Oct/images100-200')
#reference_data2  = tc.image_analysis.load_images('/data/shared/ConceptualCaptions/downloads_26Oct/images200_300')
#reference_data3  = tc.image_analysis.load_images('/data/shared/ConceptualCaptions/downloads_26Oct/images300_400')
#reference_data4  = tc.image_analysis.load_images('/data/shared/ConceptualCaptions/downloads_26Oct/images400_500')
#reference_data5  = tc.image_analysis.load_images('/data/shared/ConceptualCaptions/downloads_26Oct/images500_600')
#reference_data6  = tc.image_analysis.load_images('/data/shared/ConceptualCaptions/downloads_26Oct/images600_700')
#reference_data7  = tc.image_analysis.load_images('/data/shared/ConceptualCaptions/downloads_26Oct/images700_800')
#reference_data8  = tc.image_analysis.load_images('/data/shared/ConceptualCaptions/downloads_26Oct/images800_900')
#reference_data9  = tc.image_analysis.load_images('/data/shared/ConceptualCaptions/downloads_26Oct/images900_1000')
#reference_data0 = reference_data0.append(reference_data1)
#reference_data0 = reference_data0.append(reference_data2)
#reference_data0 = reference_data0.append(reference_data3)
#reference_data0 = reference_data0.append(reference_data4)
#reference_data0 = reference_data0.append(reference_data5)
#reference_data0 = reference_data0.append(reference_data6)
#reference_data0 = reference_data0.append(reference_data7)
#reference_data0 = reference_data0.append(reference_data8)
#reference_data0 = reference_data0.append(reference_data9)
#print('creatingg referans data')
#reference_data0 = reference_data0.add_row_number()
#pdb.set_trace()
# Save the SFrame for future use
#reference_data0.save('./conceptualcaptions_0_2.sframe')
reference_data = tc.load_sframe('./conceptualcaptions_0_2.sframe')
print('creating model')
#pdb.set_trace()
model = tc.image_similarity.create(reference_data)
model.save('./reference_data_all.model')
