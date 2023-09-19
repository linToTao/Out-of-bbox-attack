from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import os
import pybboxes as pbx
from collections import defaultdict
from tqdm import tqdm
import shutil


if __name__ == '__main__':

    imgFolder = "/pub/data/lin/COCO/train2017"
    annFile = "/pub/data/lin/COCO/annotations/instances_train2017.json"
    output_img_path = './dataset/coco/600train_stop_images/'
    output_lab_path = './dataset/coco/600train_stop_labels/'

    coco = COCO(annFile)
    catIds = coco.getCatIds(catNms=['stop sign'])
    imgIds = coco.getImgIds(catIds=catIds)

    img_dict = defaultdict(list)
    # num_imgs = len(imgIds)
    need = 600
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

                img_dict[img_name].append(line)

        if num_imgs == need:

            break
    print("The number of images is " + str(len(img_dict)))
    labels_dir = f'{output_lab_path}'
    img_dir = f'{output_img_path}'
    if os.path.exists(labels_dir):
        shutil.rmtree(labels_dir)
        shutil.rmtree(img_dir)
    os.mkdir(labels_dir)
    os.mkdir(img_dir)
    for img_name, lines in img_dict.items():
        relative_path = os.path.join(imgFolder, img_name)
        target_path = os.path.join(output_img_path, img_name)
        shutil.copy(relative_path, target_path)
        img_name = img_name.split('.')[0]
        with open(f'{labels_dir}/{img_name}.txt', 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')