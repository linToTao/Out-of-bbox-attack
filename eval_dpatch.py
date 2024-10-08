import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import argparse
is_surrogate = False
attack_mode = 'trigger'  # 'trigger', 'patch'
model_name = "yolov3"  # options : yolov3, yolov5, fasterrcnn
lr = 8
# method_num = 3
best_step = 637
test_or_train = "test"
output_mode = 1  # options:  0(training data. no-patch and label without confidence)   /   1(evalution. with-pacth and label with confidence)
print("model_name = " + model_name)
# print("method_num = " + str(method_num))
print("best_step = " + str(best_step))
print("test_or_train = " + test_or_train)
print("output_mode = " + str(output_mode))
print("lr = " + str(lr))
print("----------------------------------")
if model_name == "yolov3":

    from PyTorchYOLOv3.detect import DetectorYolov3


if model_name == "yolov5":
    from yolov5.detect import DetectorYolov5

import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from PIL import Image, ImageDraw, ImageFont
import matplotlib.image as mpimg
import time
import cv2
from tqdm import tqdm
from torch import autograd
from ensemble_tool.utils import *
from ensemble_tool.model import train_rowPtach, eval_DPtach

from adversarialYolo.demo import DetectorYolov2
from adversarialYolo.load_data import AdvDataset, PatchTransformer, PatchApplier, PatchTransformer_out_of_bbox
import count_map.main as eval_map

Gparser = argparse.ArgumentParser(description='Advpatch evaluation')
# Gparser.add_argument('--model', default='yolov4', type=str, help='options : yolov2, yolov3, yolov4, fasterrcnn')
# Gparser.add_argument('--tiny', action='store_true', help='options :True or False')
# Gparser.add_argument('--patch', default='', help='patch location')

apt1, unpar = Gparser.parse_known_args()
if attack_mode == 'patch':
    num_of_patches = 2
    patch_scale = 0.36  # patch size
if attack_mode == 'trigger':
    num_of_patches = 1
position = 'down'
cls_id_attacked = 11  # (11: stop sign  9: traffic light  46: banana  47: apple). List: https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda
if cls_id_attacked == 11:
    position = 'down'
    dataset_second = "simulator"
    bias_coordinate = 1.5  # use in ‘trigger’ attack_mode
    patch_scale = 0.64
elif cls_id_attacked == 9:
    dataset_second = "light"
    bias_coordinate = 1.6  # use in ‘trigger’ attack_mode
    patch_scale = 0.36
elif cls_id_attacked == 46:
    dataset_second = "banana"
    bias_coordinate = 1.6  # use in ‘trigger’ attack_mode
    patch_scale = 0.36
elif cls_id_attacked == 47:
    dataset_second = "apple"
    bias_coordinate = 1.6  # use in ‘trigger’ attack_mode
    patch_scale = 0.36
print("dataset is " + dataset_second)

use_FG = False
yolo_tiny = False  # only yolo54, yolov3
by_rectangle = False  # True
# transformation options
enable_rotation = False
enable_randomLocation = False
enable_crease = False
enable_projection = False
enable_rectOccluding = False
enable_blurred = False
# output images with bbox
enable_with_bbox = True  # outputs with bbox
# other setting
enable_show_plt = False  # check output images during testing by human
enable_no_random = True  # NOT randon patch "light and shadow changes"
enable_check_patch = False  # check input patch by human

max_labels_per_img = 14  # maximum number of objects per image
patch_mode = 0  # options: 0(patch), 1(white), 2(gray), 3(randon)
# fake_images_path           = "../adversarial-attack-ensemble/patch_sample/3output.png"
# fake_images_path           = "../adversarial-attack-ensemble/exp/exp07/generated/generated-images-1000.png"

# if method_num == 0:
#     method_dir = "baseline_" + model_name + "_" + str(lr)
# elif method_num == 1:
#     method_dir = model_name + "_method" + str(method_num) + "_" + str(lr)
# else:
#     method_dir = model_name + "_method" + str(method_num) + "_" + str(lr)
method_dir = model_name + "_adam_" + str(lr) + '_exp'
method_dir = model_name + '_Dpatch_exp_sim'
if attack_mode == 'patch':
    fake_images_path = ["./exp/exp2/generated/generated-images0-0004.png",
                        "./exp/exp2/generated/generated-images1-0004.png"]
elif attack_mode == 'trigger':

    fake_images_path = ["/pub/data/lin/exp/" + method_dir + "/generated/generated-images0-0" + str(best_step) + ".png"]

    # fake_images_path = ["./exp/" + method_dir + "/generated/generated-images0-0" + str(best_step) + ".png"]
    # fake_images_path = ["./exp_repeat1/" + method_dir + "/generated/generated-images0-0" + str(best_step) + ".png"]


video_name = "WIN_20210113_18_36_46_Pro"  # WIN_20200903_16_52_27_Pro, WIN_20200903_17_17_34_Pro, WIN_20210113_18_36_46_Pro
video_folder = "./dataset/video/"
source_folder = "./dataset/coco/" + test_or_train + "_stop_images/"  # "./dataset/coco/test_stop_images/", "./dataset/coco/train_stop_images/"
label_labelRescale_folder = "./dataset/coco/" + test_or_train + "_stop_yolo-labels-rescale_" + model_name
# label_labelRescale_folder = "./dataset/coco/" + test_or_train + "_stop_yolo-labels-rescale_yolov5_surrogate"
# label_labelRescale_folder = '/pub/data/lin/eval_output/surrogate/' + test_or_train + "_stop_yolo-labels-rescale_yolov5_surrogate"
# video or folder
source_key = 1  # 1:img     0:video

# MAP

if yolo_tiny == True and model_name != "yolov2":
    sss = model_name + "tiny"
else:
    if is_surrogate == False:
        sss = model_name
    else:
        sss = model_name + "_surrogate"

enable_show_map_process = False

# sss = sss+'_'+fake_images_path[35:40] # -6:-4
temp_f = fake_images_path[0].split('/')[2]
if temp_f[0] == 'exp':
    sss = sss + '_' + temp_f
else:
    # sss = sss + '_' + 'stop'
    # sss = sss + '_' + 'stop_val_5e-1_tvweight'
    if attack_mode == "trigger":
        sss = sss + '_' + 'exp_' + test_or_train + '_Dpatch_wo3'
        # if method_num == 0:
        #     sss = sss + '_' + 'exp_' + test_or_train + '_baseline_' + str(lr)
        # elif method_num == 1:
        #     sss = sss + '_' + 'exp_' + test_or_train + '_' + position + "_" + str(int(bias_coordinate*10))  # str(lr)
        # else:
        #     sss = sss + '_' + 'exp_' + test_or_train + '_' + position + "_" + str(int(bias_coordinate*10))  # str(lr)

    elif attack_mode == "patch":
        sss = model_name + "_exp_in_obj"

# st()
# output path
if output_mode == 1:
    enable_count_map = True  # False
    output_video_name = "video_output"
    output_folder = "/pub/data/lin/eval_output/" + sss + "/"  # "eval_output/" + sss + "/"
    # output_folder = "./eval_output/" + sss + "/"  # "eval_output/" + sss + "/"
    output_labels_folder = output_folder + "output_imgs/yolo-labels/"
    output_labelRescale_folder = output_folder + "output_imgs/yolo-labels-rescale/"
    output_video_folder = output_folder + "video/"
    output_imgs_folder = output_folder + "output_imgs/"

if output_mode == 0:
    enable_count_map = False
    output_video_name = "video_output"
    # output_folder = "./dataset/coco/"
    output_folder = "/pub/data/lin/eval_output/surrogate/"
    output_labels_folder = output_folder + test_or_train + "_stop_yolo-labels/"
    output_labelRescale_folder = output_folder + test_or_train + "_stop_yolo-labels-rescale/"
    output_video_folder = output_folder + "video/"
    output_imgs_folder = output_folder + "output_imgs/"
enable_output_data = True  # options:  True (output bbox labels and images (clear & rescale) and video)   /    False (only video)


# init
plt2tensor = transforms.Compose([
    transforms.ToTensor()])
device = get_default_device()

# init output folder name
tiny_str = ""
if (yolo_tiny):
    if (model_name == "yolov3" or model_name == "yolov5"):
        tiny_str = "tiny"
if (model_name == "yolov2"):
    output_labels_folder = output_labels_folder[:-1] + "_yolov2" + tiny_str + output_labels_folder[-1]
    output_labelRescale_folder = output_labelRescale_folder[:-1] + "_yolov2" + tiny_str + output_labels_folder[-1]
elif (model_name == "yolov3"):

    output_labels_folder = output_labels_folder[:-1] + "_yolov3" + tiny_str + output_labels_folder[-1]
    output_labelRescale_folder = output_labelRescale_folder[:-1] + "_yolov3" + tiny_str + output_labels_folder[-1]
elif (model_name == "yolov5"):
    if is_surrogate == False:
        output_labels_folder = output_labels_folder[:-1] + "_yolov5" + tiny_str + output_labels_folder[-1]
        output_labelRescale_folder = output_labelRescale_folder[:-1] + "_yolov5" + tiny_str + output_labels_folder[-1]
    if is_surrogate == True:
        output_labels_folder = output_labels_folder[:-1] + "_yolov5_surrogate" + tiny_str + output_labels_folder[-1]
        output_labelRescale_folder = output_labelRescale_folder[:-1] + "_yolov5_surrogate" + tiny_str + output_labels_folder[-1]

if (model_name == "fasterrcnn"):
    output_labels_folder = output_labels_folder[:-1] + "_fasterrcnn" + tiny_str + output_labels_folder[-1]
    output_labelRescale_folder = output_labelRescale_folder[:-1] + "_fasterrcnn" + tiny_str + output_labels_folder[-1]

# init cls_conf_threshold
# options:  Test (labels-rescale contain [confidence])    /    Train (labels-rescale doesn't contain [confidence])
if (output_mode == 1):
    output_data_type = "Test"
elif (output_mode == 0):
    output_data_type = "Train"
if (output_data_type == "Train"):
    cls_conf_threshold = 0.0
    enable_clear_output = True
elif (output_data_type == "Test"):
    cls_conf_threshold = 0.5
    enable_clear_output = False

# init patch_transformer and patch_applier
if torch.cuda.is_available():
    if attack_mode == 'patch':
        patch_transformer = PatchTransformer().cuda()
    elif attack_mode == 'trigger':
        patch_transformer = PatchTransformer_out_of_bbox(bias_coordinate,position=position).cuda()
    patch_applier = PatchApplier().cuda()
else:
    if attack_mode == 'patch':
        patch_transformer = PatchTransformer()
    elif attack_mode == 'trigger':
        patch_transformer = PatchTransformer_out_of_bbox(bias_coordinate, position=position)
    patch_applier = PatchApplier()

# make output folder
if (enable_output_data):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_video_folder, exist_ok=True)
    os.makedirs(output_imgs_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)
    os.makedirs(output_labelRescale_folder, exist_ok=True)


start_r = time.time()
source_data = None
fps = None
output_name = None
if (source_key == 0):
    print("Start to read images from video")
    # init
    filename = video_folder + str(video_name) + ".mp4"
    vid = imageio.get_reader(filename, 'ffmpeg')
    fps = vid.get_meta_data()['fps']
    print("input video fps : " + str(fps))
    # number of frames
    nframes = (len(list(enumerate(vid))))
    source_data = vid
    output_name = [video_name]
elif (source_key == 1):
    # read images
    print("Start to read images from folder")
    images = []
    filenames = []
    for filename in os.listdir(source_folder):
        if (filename.endswith('.jpg') or filename.endswith('.png')):
            # image = imageio.v2.imread(source_folder + filename)
            image = Image.open(source_folder + filename).convert('RGB')
            images.append(image)
            filenames.append(filename[:-4])
    # number of frames
    nframes = len(images)
    source_data = images
    output_name = filenames
finish_r = time.time()
print('Finish reading images in %f seconds.' % (finish_r - start_r))

# Read patch image
fake_images_inputs = []
for fk_img_path in fake_images_path:
    fake_images_input = Image.open(fk_img_path).convert('RGB')

    # f_width, f_height = fake_images_inputs[0].size
    # new_side = max(f_width, f_height)
    # newsize = (new_side, new_side)
    # fake_images_input = fake_images_input.resize(newsize)
    if (enable_check_patch):
        # Ckeck Images
        fake_images_input.show()
    # plt to tensor
    plt2tensor = transforms.Compose([transforms.ToTensor()])
    fake_images_input = plt2tensor(fake_images_input).unsqueeze(0)
    fake_images_input = fake_images_input.to(device, torch.float)
    fake_images_inputs.append(fake_images_input)
# if (patch_mode == 1):
#     # white
#     fake_images_input = torch.ones((3, fake_images_input.size()[-2], fake_images_input.size()[-1]), device=device).to(torch.float).unsqueeze(0)
# elif (patch_mode == 2):
#     # gray
#     fake_images_input = torch.zeros((3, fake_images_input.size()[-2], fake_images_input.size()[-1]), device=device).to(torch.float).unsqueeze(0) + 0.5
# elif (patch_mode == 3):
#     # randon
#     fake_images_input = torch.rand((3, fake_images_input.size()[-2], fake_images_input.size()[-1]), device=device).to(torch.float).unsqueeze(0)

# select detector
if (model_name == "yolov2"):
    detectorYolov2 = DetectorYolov2(show_detail=False)
    detector = detectorYolov2
if (model_name == "yolov3"):

    detectorYolov3 = DetectorYolov3(show_detail=False, tiny=yolo_tiny)

    detector = detectorYolov3
    img_size = 416
if (model_name == "yolov5"):
    if is_surrogate == False:
        detectorYolov5 = DetectorYolov5(show_detail=False)
    if is_surrogate == True:
        detectorYolov5 = DetectorYolov5(weightfile="yolov5/weight/yolov5_surrogate.pt", show_detail=False, use_FG=use_FG)
    detector = detectorYolov5
    img_size = 640
if (model_name == "fasterrcnn"):
    # just use fasterrcnn directly
    detector = None

# output video
batch_size = 1  # one by one
# if (fps == None):
#     fps = 2
# video_writer = imageio.get_writer(output_video_folder + output_video_name + ".mp4", fps=fps)
no_detect_id_list = []
low_conf_id_list = []
conf_list = []
print("\n\n")
print("================== Froward! ==================")
for i, imm in tqdm(enumerate(source_data), desc=f'Output video ', total=nframes):

    # if i>3: break
    # imm = np.asarray(imm)
    # img = Image.fromarray(imm, 'RGB')
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

    resize = transforms.Resize((img_size, img_size))
    img = resize(padded_img)  # choose here
    # to tensor
    imm_tensor = plt2tensor(img).unsqueeze(0)
    imm_tensor = imm_tensor.to(device, torch.float)
    img_side = imm_tensor.size()[-1]
    img_output = imm_tensor
    # print("imm_tensor size : "+str(imm_tensor.size()))
    # get clear label of input images
    # detect image. # Be always with bbox
    if (model_name == "yolov3" or model_name == "yolov5"):
        max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=imm_tensor,
                                                                  cls_id_attacked=cls_id_attacked, with_bbox=True)

    # add patch
    # get bbox label.
    labels = []  # format:  (label, x_center, y_center, w, h)  ex:(0 0.5 0.6 0.07 0.22)
    labels_rescale = []  # format:  (label, cls_confendence * bbox_conf, left, top, right, bottom)  ex:(person 0.76 0.6 183.1 113.5 240.3 184.7)
    if (len(bboxes) == batch_size):
        ## ONLY batch_size = 1
        bbox = bboxes[0].detach().cpu()
    if (model_name == "yolov3" or model_name == "yolov5"):
        for b in bbox:
            if (int(b[-1]) == int(cls_id_attacked)):
                label = np.array([b[-1], (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, (b[2] - b[0]), (b[3] - b[1]), b[4]],
                                 dtype=np.float32)
                labels.append(label)
                b[:-3] = b[:-3] * img_side
                label_rescale = np.array([b[-1], b[-2] * b[-3], b[0], b[1], b[2], b[3]], dtype=np.float32)
                labels_rescale.append(label_rescale)
        labels = np.array(labels)
        labels_rescale = np.array(labels_rescale)

    # Take only the top 14 largest of objectness_conf (max_labels_per_img)
    if (labels.shape[0] > 0):
        num_bbox, _ = labels.shape
        if (num_bbox > max_labels_per_img):
            # sort
            labels_sorted = labels[np.argsort(-labels[:, 5])]
            labels_rescale_sorted = labels_rescale[np.argsort(-labels[:, 5])]
            # clamp
            labels = labels_sorted[:max_labels_per_img, 0:5]
            labels_rescale = labels_rescale_sorted[:max_labels_per_img]
        else:
            labels = labels[:, 0:5]  # without conf_obj

    # set output name
    if (len(output_name) == 1):
        iname = output_name[0] + "_" + str(i)
    else:
        iname = output_name[i]

    # eval_rowPtach
    if (len(labels) > 0):
        labels_tensor = plt2tensor(labels).to(device)
        no_detect_img = False
        low_conf_img = False
        p_img_batch, fake_images_denorm, bboxes, no_detect_img, low_conf_img = eval_DPtach(generator=None,
                                                                                             batch_size=batch_size,
                                                                                             device=device,
                                                                                             latent_shift=None,
                                                                                             alpah_latent=None,
                                                                                             input_imgs=imm_tensor,
                                                                                             label=labels_tensor,
                                                                                             patch_scale=patch_scale,
                                                                                             cls_id_attacked=cls_id_attacked,
                                                                                             denormalisation=False,
                                                                                             model_name=model_name,
                                                                                             detector=detector,
                                                                                             patch_transformer=patch_transformer,
                                                                                             patch_applier=patch_applier,
                                                                                             by_rectangle=by_rectangle,
                                                                                             enable_rotation=enable_rotation,
                                                                                             enable_randomLocation=enable_randomLocation,
                                                                                             enable_crease=enable_crease,
                                                                                             enable_projection=enable_projection,
                                                                                             enable_rectOccluding=enable_rectOccluding,
                                                                                             enable_blurred=enable_blurred,
                                                                                             enable_with_bbox=enable_with_bbox,
                                                                                             enable_show_plt=enable_show_plt,
                                                                                             enable_clear_output=enable_clear_output,
                                                                                             cls_conf_threshold=cls_conf_threshold,patch_mode=patch_mode,
                                                                                             enable_no_random=enable_no_random,
                                                                                             fake_images_default=fake_images_inputs,attack_mode=attack_mode,
                                                                                             img_size=img_size,
                                                                                             no_detect_img=no_detect_img,
                                                                                             low_conf_img=low_conf_img,
                                                                                             conf_list=conf_list)

        img_output = p_img_batch

        if low_conf_img:
            low_conf_id_list.append(i)
        if not (enable_clear_output):
            # get bbox label.
            labels = []  # format:  (label, x_center, y_center, w, h)  ex:(0 0.5 0.6 0.07 0.22)
            labels_rescale = []  # format:  (label, confendence, left, top, right, bottom)  ex:(person 0.76 0.6 183.1 113.5 240.3 184.7)
            if (len(bboxes) == batch_size):
                ## ONLY batch_size = 1
                bbox = bboxes[0].detach().cpu()
            if (model_name == "yolov3" or model_name == "yolov5" or model_name == "fasterrcnn"):
                for b in bbox:
                    if (int(b[-1]) == int(cls_id_attacked)):
                        label = np.array(
                            [b[-1], (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, (b[2] - b[0]), (b[3] - b[1]), b[4]],
                            dtype=np.float32)
                        labels.append(label)
                        b[:-3] = b[:-3] * img_side
                        label_rescale = np.array([b[-1], b[-2] * b[-3], b[0], b[1], b[2], b[3]], dtype=np.float32)
                        labels_rescale.append(label_rescale)
                labels = np.array(labels)
                labels_rescale = np.array(labels_rescale)
            elif (model_name == "yolov2"):
                for b in bbox:
                    if (int(b[-1]) == int(cls_id_attacked)):
                        label = np.array([b[-1], b[0], b[1], b[2], b[3], b[4]], dtype=np.float32)
                        labels.append(label)
                        b[:-3] = b[:-3] * img_side
                        label_rescale = np.array(
                            [b[-1], b[-2] * b[-3], (b[0] - (b[2] / 2.0)), (b[1] - (b[3] / 2.0)), (b[0] + (b[2] / 2.0)),
                             (b[1] + (b[3] / 2.0))], dtype=np.float32)
                        labels_rescale.append(label_rescale)
                labels = np.array(labels)
                labels_rescale = np.array(labels_rescale)
            # Take only the top 14 largest of objectness_conf (max_labels_per_img)
            if (labels.shape[0] > 0):
                num_bbox, _ = labels.shape
                if (num_bbox > max_labels_per_img):
                    # sort
                    labels_sorted = labels[np.argsort(-labels[:, 5])]
                    labels_rescale_sorted = labels_rescale[np.argsort(-labels[:, 5])]
                    # clamp
                    labels = labels_sorted[:max_labels_per_img, 0:5]
                    labels_rescale = labels_rescale_sorted[:max_labels_per_img]
                else:
                    labels = labels[:, 0:5]  # without conf_obj

    # output data
    if (enable_output_data):
        # save clear imgs
        output_path = str(output_imgs_folder) + '%s.png' % (iname)
        save_image(img_output, output_path)
        # save bbox
        output_path = str(output_labels_folder) + '%s.txt' % (iname)
        np.savetxt(output_path, labels, fmt='%.6f')
    if (enable_output_data):
        # save recale bbox
        output_path = output_labelRescale_folder + iname + ".txt"
        labelfile_rescale = open(output_path, 'w+')  # read label
        for bbox in labels_rescale:
            if (output_data_type == "Train"):
                labelfile_rescale.write(
                    "stop" + str(f' {bbox[2]} {bbox[3]} {bbox[4]} {bbox[5]}\n'))  # left, top, right, bottom
            elif (output_data_type == "Test"):
                labelfile_rescale.write("stop" + str(
                    f' {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]} {bbox[5]}\n'))  # confendence left, top, right, bottom
        labelfile_rescale.close()
        # output video
#     img_output = img_output[0].cpu().detach().numpy()
#     img_output = np.transpose(img_output, (1, 2, 0))
#     img_output = 255 * img_output  # Now scale by 255
#     img_output = img_output.astype(np.uint8)
#     video_writer.append_data(img_output)
# video_writer.close()
print("\n\n")

# output video
# print("================ output video ================")
# if (fps == None):
#     fps = 2
# # video_writer = imageio.get_writer(output_video_folder + output_video_name + ".mp4", fps=fps)
# video_size = (416, 416)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video_writer = cv2.VideoWriter(output_video_folder + output_video_name + ".mp4",  fourcc, fps, video_size, True)

# for filename in tqdm(os.listdir(output_imgs_folder), desc=f'Output video ', total=nframes):
#     if (filename.endswith('.jpg') or filename.endswith('.png')):

#         image = cv2.imread(output_imgs_folder + filename)
#         video_writer.write(image)

# video_writer.release()
# print("\n\n")

# st()
# MAP
if (enable_count_map):
    if not (enable_show_map_process):
        output_imgs_folder = None
    # st()
    output_map = eval_map.count(path_ground_truth=label_labelRescale_folder,
                                path_detection_results=output_labelRescale_folder,
                                path_images_optional=output_imgs_folder)
    # save
    # with open("./"+output_folder+"map.txt", "w") as text_file:
    #     text_file.write(str(output_map))

print(fake_images_path)
if yolo_tiny == True and model_name != 'yolov2':
    model_name = model_name + '_tiny'
print(model_name)
print("\n\n")

print(output_labels_folder)
folder_path = output_labelRescale_folder
n = 0
if (enable_count_map):
    # files = os.listdir("../" + folder_path)
    files = os.listdir(folder_path)
else:
    files = os.listdir(folder_path)
for file_name in files:
    ab_name = folder_path + file_name
    if (enable_count_map):
        # fsize = os.path.getsize("../" + ab_name)
        fsize = os.path.getsize(ab_name)
    else:
        fsize = os.path.getsize(ab_name)
    if fsize == 0:
        n = n + 1
print("The number of empty file is " + str(n))
print('=================== finish ===================\n\n')
