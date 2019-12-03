import turicreate as tc
tc.config.set_num_gpus(0)
import pdb
# Load images from the downloaded data
reference_data  = tc.image_analysis.load_images('/Users/emre/Desktop/images/')
reference_data1  = tc.image_analysis.load_images('/Users/emre/Desktop/images1/')
#print('creatingg referans data')
#reference_data = reference_data.add_row_number()

# Save the SFrame for future use
#reference_data.save('./conceptualcaptions.sframe')
# reference_data = tc.load_sframe('./conceptualcaptions.sframe')
print('creating model')
# pdb.set_trace()
model = tc.image_similarity.create(reference_data)
model.save('./5.model')
query = model.query(reference_data1[0:1], k=5)