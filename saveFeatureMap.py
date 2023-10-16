import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys
import random
import argparse

model_name = "yolov3"  # options : yolov3, yolov5, fasterrcnn
if model_name == "yolov3":
    from PyTorchYOLOv3.detect import DetectorYolov3
    print("Victim model is yolov3")
if model_name == "yolov5":
    from yolov5.detect import DetectorYolov5
    print("Victim model is yolov5")

import numpy as np
from tqdm import tqdm
from torch import autograd
from torch.utils.data import DataLoader
from ensemble_tool.utils import *
from ensemble_tool.model import train_rowPtach, TotalVariation, IFGSM
# from GANLatentDiscovery.loading import load_from_dir
# from GANLatentDiscovery.utils import is_conditional
# from pytorch_pretrained_detection import FasterrcnnResnet50, MaskrcnnResnet50
# from pytorchYOLOv4.demo import DetectorYolov4
# from adversarialYolo.demo import DetectorYolov2
# from adversarialYolo.train_patch import PatchTrainer
from adversarialYolo.load_data import AdvDataset, PatchTransformer, PatchApplier, PatchTransformer_out_of_bbox
from pathlib import Path
import fnmatch
import shutil


grey_mask_img_path = './dataset/coco/train_stop_images_withMask/'
output_feature_path = './dataset/coco/train_image_feature-' + model_name

if os.path.exists(output_feature_path):
    shutil.rmtree(output_feature_path)
os.mkdir(output_feature_path)

plt2tensor = transforms.Compose([
    transforms.ToTensor()])
device = get_default_device()
# mask_img_list = fnmatch.filter(os.listdir(grey_mask_img_path), '*.png') + fnmatch.filter(os.listdir(grey_mask_img_path), '*.jpg')

if model_name == "yolov3":
    yolo_tiny = False
    detectorYolov3 = DetectorYolov3(show_detail=False, tiny=yolo_tiny)
    detector = detectorYolov3
    batch_size_second = 16
    cls_conf_threshold = 0.
    ds_image_size_second = 416
    # learing_rate = 0.005
    if yolo_tiny == False:
        batch_size_second = 16
    features_in_hook = []
    def hook(module, fea_in, fea_out):
        # print("hooker working")
        features_in_hook.append(fea_in[0])
        # features_out_hook.append(fea_out[0])
        return None
    model = detector.model
    model_list = model.module_list
    # print(model_list)
    for sequential in model_list:
        for name, layer in sequential._modules.items():
            # if name == "conv_37":
            #     layer.register_forward_hook(hook=hook)
            #     print("Hook conv_37     done!!!")
            # if name == "conv_62":
            #     layer.register_forward_hook(hook=hook)
            #     print("Hook conv_62     done!!!")
            # if name == "conv_75":
            #     layer.register_forward_hook(hook=hook)
            #     print("Hook conv_75     done!!!")
            if name == "conv_80":
                layer.register_forward_hook(hook=hook)
                print("Hook conv_80     done!!!")
            if name == "conv_92":
                layer.register_forward_hook(hook=hook)
                print("Hook conv_92     done!!!")
            if name == "conv_104":
                layer.register_forward_hook(hook=hook)
                print("Hook conv_104    done!!!")

    images = []
    filenames = []
    for filename in os.listdir(grey_mask_img_path):
        if (filename.endswith('.jpg') or filename.endswith('.png')):
            # image = imageio.v2.imread(source_folder + filename)
            image = Image.open(grey_mask_img_path + filename).convert('RGB')
            images.append(image)
            filenames.append(filename[:-4])
    nframes = len(images)
    source_data = images
    output_name = filenames

    for i, imm in tqdm(enumerate(source_data), desc=f'Output feature', total=nframes):
        img = imm
        w, h = img.size
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))

            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))

        resize = transforms.Resize((416, 416))
        img = resize(padded_img)  # choose here
        # to tensor
        imm_tensor = plt2tensor(img).unsqueeze(0)
        imm_tensor = imm_tensor.to(device, torch.float)
        img_side = imm_tensor.size()[-1]

        model(imm_tensor)
        for id, feature in enumerate(features_in_hook):
            vector_fname = output_name[i] + '-' + str(id) + '.pt'
            torch.save(feature.cpu().detach(), os.path.join(output_feature_path, vector_fname))
            # print(feature.cpu().detach().size())
            feature.cpu().detach()
        features_in_hook = []

if model_name == "yolov5":
    detectorYolov5 = DetectorYolov5(show_detail=False)
    detector = detectorYolov5
    batch_size_second = 16
    cls_conf_threshold = 0.
    ds_image_size_second = 640
    # learing_rate = 0.005
    features_in_hook = []
    def hook(module, fea_in, fea_out):
        # print("hooker working")
        features_in_hook.append(fea_in[0])
        # features_out_hook.append(fea_out[0])
        return None
    model = detector.model.model.model
    for name, layer in model._modules.items():
        if name == "24":
            for hook_name, hook_layer in layer.m._modules.items():
                hook_layer.register_forward_hook(hook=hook)
                print("Hook " + str((hook_name, hook_layer)) + " done!!!")

num_files = len(os.listdir(output_feature_path))
print(num_files)