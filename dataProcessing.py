from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import os
import pybboxes as pbx
from collections import defaultdict
from tqdm import tqdm
import shutil
import cv2
from PIL import Image

if __name__ == '__main__':

    imgFolder = "/pub/data/lin/COCO/train2017"
    annFile = "/pub/data/lin/COCO/annotations/instances_train2017.json"
    output_train_img_path = './dataset/coco/train_stop_images/'
    output_train_lab_path = './dataset/coco/train_stop_labels/'
    output_test_img_path = './dataset/coco/test_stop_images/'
    output_test_lab_path = './dataset/coco/test_stop_labels/'
    output_train_with_grey_mask_img_path = './dataset/coco/train_stop_images_withGrayMask/'

    coco = COCO(annFile)
    catIds = coco.getCatIds(catNms=['stop sign'])
    imgIds = coco.getImgIds(catIds=catIds)

    train_img_dict = defaultdict(list)
    test_img_dict = defaultdict(list)

    # num_imgs = len(imgIds)

    num_imgs = 0
    for idx in tqdm(range(len(imgIds))):
        imgid = imgIds[idx]
        # print(imgid)
        img = coco.loadImgs(imgid)[0]
        # print(img['id'])
        anno_id = coco.getAnnIds(imgIds=img['id'], catIds=13, iscrowd=None)

        anns = coco.loadAnns(anno_id)

        img_name = img['file_name']
        W, H = img['width'], img['height']

        is_good_img = False
        for ann in anns:
            if 0.01 <= ((ann['bbox'][2] * ann['bbox'][3]) / (W * H)):
                is_good_img = True
                break
        if is_good_img:
            num_imgs += 1
            if num_imgs <= 600:
                for ann in anns:
                    sample_label_list = []

                    coco_bbox = tuple(ann['bbox'])
                    x_center, y_center, w, h = pbx.convert_bbox(coco_bbox, from_type="coco", to_type="yolo",
                                                                image_size=(W, H))
                    class_num = 11

                    sample_label_list.append(str(class_num))
                    sample_label_list.append(str(x_center))
                    sample_label_list.append(str(y_center))
                    sample_label_list.append(str(w))
                    sample_label_list.append(str(h))
                    line = ' '.join(sample_label_list)

                    train_img_dict[img_name].append(line)

            if num_imgs > 600:
                if num_imgs > 1100:
                    break
                for ann in anns:
                    sample_label_list = []

                    coco_bbox = tuple(ann['bbox'])
                    x_center, y_center, w, h = pbx.convert_bbox(coco_bbox, from_type="coco", to_type="yolo",
                                                                image_size=(W, H))
                    class_num = 11

                    sample_label_list.append(str(class_num))
                    sample_label_list.append(str(x_center))
                    sample_label_list.append(str(y_center))
                    sample_label_list.append(str(w))
                    sample_label_list.append(str(h))
                    line = ' '.join(sample_label_list)

                    test_img_dict[img_name].append(line)

    print("The number of train_images is " + str(len(train_img_dict)))
    print("The number of test_images is " + str(len(test_img_dict)))

    train_labels_dir = f'{output_train_lab_path}'
    train_img_dir = f'{output_train_img_path}'
    test_labels_dir = f'{output_test_lab_path}'
    test_img_dir = f'{output_test_img_path}'
    if os.path.exists(train_labels_dir):
        shutil.rmtree(train_labels_dir)
    if os.path.exists(train_img_dir):
        shutil.rmtree(train_img_dir)
    if os.path.exists(test_labels_dir):
        shutil.rmtree(test_labels_dir)
    if os.path.exists(test_img_dir):
        shutil.rmtree(test_img_dir)
    if os.path.exists(output_train_with_grey_mask_img_path):
        shutil.rmtree(output_train_with_grey_mask_img_path)
    os.mkdir(output_train_with_grey_mask_img_path)
    os.mkdir(train_labels_dir)
    os.mkdir(train_img_dir)
    os.mkdir(test_labels_dir)
    os.mkdir(test_img_dir)

    for img_name, lines in train_img_dict.items():
        relative_path = os.path.join(imgFolder, img_name)
        target_path = os.path.join(output_train_img_path, img_name)
        shutil.copy(relative_path, target_path)
        img_name = img_name.split('.')[0]
        with open(f'{train_labels_dir}/{img_name}.txt', 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')

    for img_name, lines in test_img_dict.items():
        relative_path = os.path.join(imgFolder, img_name)
        target_path = os.path.join(output_test_img_path, img_name)
        shutil.copy(relative_path, target_path)
        img_name = img_name.split('.')[0]
        with open(f'{test_labels_dir}/{img_name}.txt', 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')


    def polygons_to_mask(img_shape, polygons):
        masks = []

        for polygon in polygons:
            mask = np.zeros(img_shape, dtype=np.uint8)
            polygons = np.asarray([polygon], np.int32)  # 这里必须是int32，其他类型使用fillPoly会报错
            shape = polygons.shape
            polygons = polygons.reshape(shape[0], -1, 2)
            cv2.fillPoly(mask, polygons, color=(1, 1, 1))
            masks.append(mask)
        return masks


    file_names = os.listdir(output_train_img_path)
    file_ids = [int(item.split(".")[0]) for item in file_names if item.split(".")[0] != '']
    print("Generating the images with grey mask...")
    for i in tqdm(range(len(file_ids))):
        img = coco.loadImgs(file_ids[i])[0]
        anno_id = coco.getAnnIds(imgIds=file_ids[i], catIds=13, iscrowd=None)
        anns = coco.loadAnns(anno_id)
        file_name = img['file_name']

        image = cv2.imread(output_train_img_path + file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_shape = image.shape
        masks_all = np.zeros(image.shape)
        img_mask = np.zeros(image.shape)

        for ann in anns:
            # masks_per_ann = polygons_to_mask(img_shape, ann['segmentation'])
            x1 = int(anns['bbox'][0])
            y2 = int(anns['bbox'][1] + anns['bbox'][3])
            x2 = int(anns['bbox'][0] + anns['bbox'][2])
            y1 = int(anns['bbox'][1])
            masks_all[y1:y2, x1:x2, :] = 1

            # for mask in masks_per_ann:
            #     masks_all = masks_all + mask
        mask = np.random.randint(0, 256, size=image.shape)
        img_mask = np.where((masks_all == 0), image, 128)

        # img_mask = np.where((masks_all == 0), image, mask)
        img_mask = Image.fromarray(np.uint8(img_mask))
        img_mask.save(output_train_with_grey_mask_img_path + img['file_name'])
    print("Done !!!")
    print("________________________________________")
    print("Generating the images with grey mask...")
