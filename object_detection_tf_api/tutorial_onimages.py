import numpy as np
import time
import cPickle as pickle
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from os import listdir
from os.path import isfile, join
import six.moves.urllib as urllib
import tarfile
import os
os.environ['PYTHONPATH'] += ':/home/luchy/Desktop/tensorflow/models/research/:/home/luchy/Desktop/tensorflow/models/research/slim'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# import json, codecs
# import zipfile
# from collections import defaultdict
# from io import StringIO
# from matplotlib import pyplot as plt

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'

# def save_model(model_name):
#   base_url = 'http://download.tensorflow.org/models/object_detection/'
#   model_file = model_name + '.tar.gz'
#   model_dir = tf.keras.utils.get_file(
#     fname=model_name,
#     origin=base_url + model_file,
#     untar=True)
# save_model(model_name)


#base path where we will save our models
PATH_TO_OBJ_DETECTION = '/home/luchy/.keras/datasets'
# Specify Model To Download. Obtain the model name from the object detection model zoo.
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
# MODEL_NAME = 'ssd_resnet101_v1_fpn_shared_box_predictor_oid_512x512_sync_2019_01_20'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

DESTINATION_MODEL_TAR_PATH = PATH_TO_OBJ_DETECTION + '/data/' + MODEL_FILE
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, DESTINATION_MODEL_TAR_PATH)
# tar_file = tarfile.open(DESTINATION_MODEL_TAR_PATH)
# for file in tar_file.getmembers():
#     file_name = os.path.basename(file.name)
#     if 'frozen_inference_graph.pb' in file_name:
#         tar_file.extract(file, PATH_TO_OBJ_DETECTION+'/data')

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = PATH_TO_OBJ_DETECTION + '/data/' + MODEL_NAME + '/frozen_inference_graph.pb'
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/luchy/Desktop/tensorflow/models/research/object_detection/data/mscoco_label_map.pbtxt'
# PATH_TO_LABELS = '/home/luchy/Desktop/tensorflow/models/research/object_detection/data/oid_v4_label_map.pbtxt'
NUM_CLASSES = 90
# NUM_CLASSES = 601
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes',
                        'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)
                # END if tensor_name in
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
                # END IF DETECTION MASKS

            # END FOR KEY LOOP

            image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def Run_Object_Detection_On_Images(images_path, count, dir_list):
    IMAGE_SIZE = (12, 8)
    for image_path in images_path:
        if (image_path.split('/')[6].strip('.jpg') + '.pkl') in dir_list:
            continue
        print(image_path.split('/')[6])
        image = Image.open(image_path).convert('RGB')
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        # image_name = image_path.strip('/home/luchy/Desktop/images/').strip('.jpg')
        image_name = image_path.split('/')[6].strip('.jpg')
        out_path = '/media/luchy/E848C43F48C40DEE/ConCap/concap_pkl/'
        # feature_dict.update({image_name: output_dict})
        with (open(out_path + image_name +'.pkl', 'w')) as f:
            pickle.dump(output_dict, f)
        # # Visualization of the results of a detection.
        # vis_util.visualize_boxes_and_labels_on_image_array(
        #   image_np,
        #   output_dict['detection_boxes'],
        #   output_dict['detection_classes'],
        #   output_dict['detection_scores'],
        #   category_index,
        #   instance_masks=output_dict.get('detection_masks'),
        #   use_normalized_coordinates=True,
        #   line_thickness=5)
        # plt.figure(figsize=IMAGE_SIZE)
        # plt.imshow(image_np)
        # count +=1
        # name = str(count) + '.jpg'
        # plt.savefig(name)


# TEST_IMAGES_BASE_PATH = "/home/luchy/Desktop/images/"
TEST_IMAGES_BASE_PATH = "/media/luchy/E848C43F48C40DEE/ConCap/images0_100/"
TEST_IMAGES_PATHS = [TEST_IMAGES_BASE_PATH+f for f in listdir(TEST_IMAGES_BASE_PATH) if isfile(join(TEST_IMAGES_BASE_PATH, f))]

print ('How many pics:%d '%len(TEST_IMAGES_PATHS))
count = 0
feature_dict = {}
start = time.time()
dir_list = os.listdir('/media/luchy/E848C43F48C40DEE/ConCap/concap_pkl/')
Run_Object_Detection_On_Images(TEST_IMAGES_PATHS, count, dir_list)
print (time.time() - start)


# with (open('file.pkl', 'w')) as f:
#     pickle.dump(feature_dict, f)
print('Done')

##this part visualizes
# from matplotlib import pyplot as plt
# dir_list = os.listdir('/media/luchy/E848C43F48C40DEE/ConCap/concap_pkl/')
# for i in dir_list:
#     try:
#         i = i.strip('.pkl')
#         pkl_name = '/media/luchy/E848C43F48C40DEE/ConCap/concap_pkl/' + str(i) + '.pkl'
#         image_p = '/media/luchy/E848C43F48C40DEE/ConCap/images0_100/' + str(i) + '.jpg'
#         with (open(pkl_name, 'r')) as f:
#             output_dict = pickle.load(f)
#         image = Image.open(image_p)
#         image_np = load_image_into_numpy_array(image)
#         IMAGE_SIZE = (12, 8)
#         # Visualization of the results of a detection.
#         vis_util.visualize_boxes_and_labels_on_image_array(
#             image_np,
#             output_dict['detection_boxes'],
#             output_dict['detection_classes'],
#             output_dict['detection_scores'],
#             category_index,
#             instance_masks=output_dict.get('detection_masks'),
#             use_normalized_coordinates=True,
#             line_thickness=5)
#         plt.figure(figsize=IMAGE_SIZE)
#         plt.imshow(image_np)
#         count = 1000000 + int(i)
#         name = str(count) + '.jpg'
#         plt.savefig('/media/luchy/E848C43F48C40DEE/ConCap/done_images/' + name)
#         plt.close()
#     except Exception as e:
#         print(i)
#         print(e)
#         continue