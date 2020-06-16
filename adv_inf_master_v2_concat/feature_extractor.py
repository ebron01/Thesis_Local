"""
use /usr/bin/python3.6 feature_extractor.py
lib installed with
pip3 install resnet_pytorch

"""
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from resnet_pytorch import ResNet
import numpy as np
import json

model = ResNet.from_pretrained('resnet152')
# if torch.cuda.is_available():
    # model.to("cuda")

# Preprocess image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_path = '/home/luchy/Desktop/download_concap/images'
out_tensor_save_path = '/home/luchy/Desktop/features_15June'
if not os.path.exists(out_tensor_save_path):
    os.makedirs(out_tensor_save_path)
# inputs_batch = torch.zeros(len(os.listdir(image_path)), 3, 224, 224).to("cuda")
# features_batch = torch.zeros(len(os.listdir(image_path)), 2048, 1, 1)
for k, file in enumerate(os.listdir(image_path)):
    # if torch.cuda.is_available():
    print (file)
    try:
        input_image = Image.open(os.path.join(image_path, file))
    except:
        print('Cannot open image', file)
        continue
    try:
        input_tensor = preprocess(input_image)
    except:
        print('Cannot process image', file)
        continue
    # inputs = inputs.to("cuda")
    # inputs_batch[k] = input_tensor
    out_tensor = model.extract_features(input_tensor.unsqueeze(0))
    np.save(os.path.join(out_tensor_save_path, file.split('.')[0] + '.npy'), out_tensor.detach().numpy())
    # torch.save(out_tensor, os.path.join(out_tensor_save_path, file.split('.')[0] + '.pt'))
# features = model.extract_features(inputs_batch)
# print(features.shape) # torch.Size([1, 512, 1, 1])


not_parsed_cc_keys = json.load(open('/home/luchy/PycharmProjects/Thesis_Local/adv_inf_master_v2_gen/parsers/not_parsed_cc_keys.json', 'r'))
img_feat_size = 2048

print(len(not_parsed_cc_keys.keys()))
for key in not_parsed_cc_keys.keys():
    array = not_parsed_cc_keys[key]
    array_len = len(array)
    out_array = np.zeros([array_len, 1, img_feat_size], dtype='float32')
    for i in range(len(array)):
        #np.zeros([batch_size, self.max_sent_num, self.max_seg, self.opt.img_feat_size], dtype='float32')
        if not os.path.exists(os.path.join(out_tensor_save_path, array[i] + '.npy')):
            continue
        out_array[i, 0] = np.load(os.path.join(out_tensor_save_path, array[i] + '.npy'))[0, :, 0, 0]
    np.save(os.path.join('/data/shared/ActivityNet/advinf_activitynet/feats/resnet152_concap/', key + '.npy'), out_array)
    print(key + '.npy done')

print('Done')
