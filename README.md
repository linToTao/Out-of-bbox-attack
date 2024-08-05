# Out-of-Bounding-Box Triggers: A Stealthy Approach to Cheat Object Detector

Official code release for [Out-of-Bounding-Box Triggers: A Stealthy Approach to Cheat Object Detector](https://github.com/linToTao/Out-of-bbox-attack).

<p align='center'>
  <b>
    <a href="https://github.com/linToTao/Out-of-bbox-attack">Paper</a>
    |
    <a href="https://github.com/linToTao/Out-of-bbox-attack">Code</a> 
  </b>
</p> 
  <p align='center'>
    <img src='static/out-of-bbox-attack.png' width='1000'/>
  </p>

**Abstract**: In recent years, the study of adversarial robustness in object detection systems, particularly those based on deep neural networks (DNNs), has become a pivotal area of research. Traditional physical attacks targeting object detectors, such as adversarial patches and texture manipulations, directly manipulate the surface of the object. While these methods are effective, their overt manipulation of objects may draw attention in real-world applications. To address this, this paper introduces a more subtle approach: an inconspicuous adversarial trigger that operates outside the bounding boxes, rendering the object undetectable to the model. We further enhance this approach by proposing the <mark>Feature Guidance (FG)</mark> technique and the <mark>Universal Auto-PGD (UAPGD)</mark> optimization strategy for crafting high-quality triggers. The effectiveness of our method is validated through extensive empirical testing, demonstrating its high performance in both digital and physical environments.

## Getting Started

### Environmental Setups

```bash
git clone https://github.com/linToTao/Out-of-bbox-attack
cd Out-of-bbox-attack
conda env create -f environment.yml -n OOBA
conda activate OOBA
```

### Model and Dataset Download

- We can download [2017 train images](http://images.cocodataset.org/zips/train2017.zip), [2017 val images](http://images.cocodataset.org/zips/val2017.zip) and their [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) of COCO2017 from the official website.
Please create a new folder named "dataset", download and unzip the two datasets into the folder. And then we can process the data.

  ```bash
  mkdir dataset/COCO
  cd dataset/COCO
  # download and move the zip files into the folder
  ```
- download [YOLOv5m](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt) weight
  
    ```bash
    mkdir ./yolov5/weight/
    cd ./yolov5/weight/
    wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt
    ```

- download YOLOv3
  
    ```bash
    bash ./PyTorchYOLOv3/weights/download_weights.sh
    ```

### Preparation before training

- process data

    ```bash
    python ./dataProcessing.py
    ```

- save feature maps that FG method need

    ```bash
    python ./saveFeatureMap.py --model_name yolov3
    python ./saveFeatureMap.py --model_name yolov5
    ```

Finally the dateset file structure is organized as:

```
Out-of-bbox-attack
├── dataset
│   ├── COCO
│   |   ├── train2017
│   |   ├── val2017
│   |   ├── annotations
│   ├── coco
│   |   ├── train_stop_images
│   |   ├── train_stop_labels
│   |   ├── test_stop_images
│   |   ├── test_stop_labels
│   |   ├── train_stop_images_withGrayMask 
│   |   ├── train_stop_feature-yolov3
│   |   ├── train_stop_feature-yolov5
└── other codes...
```

### Train

```bash 
python ./train.py
```

### evaluation

```bash 
python ./evaluation.py
```
  
