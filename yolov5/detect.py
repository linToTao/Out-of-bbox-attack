# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import datetime
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
sys.path.append('yolov5/')
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from .models.common import DetectMultiBackend
from .utils.general import non_max_suppression


# @smart_inference_mode()
# def run(
#         weights=ROOT / 'yolov5s.pt',  # model path or triton URL
#         source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
#         data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
#         imgsz=(640, 640),  # inference size (height, width)
#         conf_thres=0.25,  # confidence threshold
#         iou_thres=0.45,  # NMS IOU threshold
#         max_det=1000,  # maximum detections per image
#         device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
#         view_img=False,  # show results
#         save_txt=False,  # save results to *.txt
#         save_csv=False,  # save results in CSV format
#         save_conf=False,  # save confidences in --save-txt labels
#         save_crop=False,  # save cropped prediction boxes
#         nosave=False,  # do not save images/videos
#         classes=None,  # filter by class: --class 0, or --class 0 2 3
#         agnostic_nms=False,  # class-agnostic NMS
#         augment=False,  # augmented inference
#         visualize=False,  # visualize features
#         update=False,  # update all models
#         project=ROOT / 'runs/detect',  # save results to project/name
#         name='exp',  # save results to project/name
#         exist_ok=False,  # existing project/name ok, do not increment
#         line_thickness=3,  # bounding box thickness (pixels)
#         hide_labels=False,  # hide labels
#         hide_conf=False,  # hide confidences
#         half=False,  # use FP16 half-precision inference
#         dnn=False,  # use OpenCV DNN for ONNX inference
#         vid_stride=1,  # video frame-rate stride
# ):
#     source = str(source)
#     save_img = not nosave and not source.endswith('.txt')  # save inference images
#     is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
#     is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
#     webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
#     screenshot = source.lower().startswith('screen')
#     if is_url and is_file:
#         source = check_file(source)  # download
#
#     # Directories
#     save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
#     (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
#
#     # Load model
#     device = select_device(device)
#     model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
#     stride, names, pt = model.stride, model.names, model.pt
#     imgsz = check_img_size(imgsz, s=stride)  # check image size
#
#     # Dataloader
#     bs = 1  # batch_size
#     if webcam:
#         view_img = check_imshow(warn=True)
#         dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
#         bs = len(dataset)
#     elif screenshot:
#         dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
#     else:
#         dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
#     vid_path, vid_writer = [None] * bs, [None] * bs
#
#     # Run inference
#     model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
#     seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
#     for path, im, im0s, vid_cap, s in dataset:
#         with dt[0]:
#             im = torch.from_numpy(im).to(model.device)
#             im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
#             im /= 255  # 0 - 255 to 0.0 - 1.0
#             if len(im.shape) == 3:
#                 im = im[None]  # expand for batch dim
#
#         # Inference
#         with dt[1]:
#             visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
#             pred = model(im, augment=augment, visualize=visualize)
#
#         # NMS
#         with dt[2]:
#             pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
#
#         # Second-stage classifier (optional)
#         # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
#
#         # Define the path for the CSV file
#         csv_path = save_dir / 'predictions.csv'
#
#         # Create or append to the CSV file
#         def write_to_csv(image_name, prediction, confidence):
#             data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
#             with open(csv_path, mode='a', newline='') as f:
#                 writer = csv.DictWriter(f, fieldnames=data.keys())
#                 if not csv_path.is_file():
#                     writer.writeheader()
#                 writer.writerow(data)
#
#         # Process predictions
#         for i, det in enumerate(pred):  # per image
#             seen += 1
#             if webcam:  # batch_size >= 1
#                 p, im0, frame = path[i], im0s[i].copy(), dataset.count
#                 s += f'{i}: '
#             else:
#                 p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
#
#             p = Path(p)  # to Path
#             save_path = str(save_dir / p.name)  # im.jpg
#             txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
#             s += '%gx%g ' % im.shape[2:]  # print string
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#             imc = im0.copy() if save_crop else im0  # for save_crop
#             annotator = Annotator(im0, line_width=line_thickness, example=str(names))
#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
#
#                 # Print results
#                 for c in det[:, 5].unique():
#                     n = (det[:, 5] == c).sum()  # detections per class
#                     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
#
#                 # Write results
#                 for *xyxy, conf, cls in reversed(det):
#                     c = int(cls)  # integer class
#                     label = names[c] if hide_conf else f'{names[c]}'
#                     confidence = float(conf)
#                     confidence_str = f'{confidence:.2f}'
#
#                     if save_csv:
#                         write_to_csv(p.name, label, confidence_str)
#
#                     if save_txt:  # Write to file
#                         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
#                         with open(f'{txt_path}.txt', 'a') as f:
#                             f.write(('%g ' * len(line)).rstrip() % line + '\n')
#
#                     if save_img or save_crop or view_img:  # Add bbox to image
#                         c = int(cls)  # integer class
#                         label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
#                         annotator.box_label(xyxy, label, color=colors(c, True))
#                     if save_crop:
#                         save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
#
#             # Stream results
#             im0 = annotator.result()
#             if view_img:
#                 if platform.system() == 'Linux' and p not in windows:
#                     windows.append(p)
#                     cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
#                     cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
#                 cv2.imshow(str(p), im0)
#                 cv2.waitKey(1)  # 1 millisecond
#
#             # Save results (image with detections)
#             if save_img:
#                 if dataset.mode == 'image':
#                     cv2.imwrite(save_path, im0)
#                 else:  # 'video' or 'stream'
#                     if vid_path[i] != save_path:  # new video
#                         vid_path[i] = save_path
#                         if isinstance(vid_writer[i], cv2.VideoWriter):
#                             vid_writer[i].release()  # release previous video writer
#                         if vid_cap:  # video
#                             fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                         else:  # stream
#                             fps, w, h = 30, im0.shape[1], im0.shape[0]
#                         save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
#                         vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#                     vid_writer[i].write(im0)
#
#         # Print time (inference-only)
#         LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
#
#     # Print results
#     t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
#     LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
#     if save_txt or save_img:
#         s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
#         LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
#     if update:
#         strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
#
#
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
#     parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
#     parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
#     parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
#     parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
#     parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='show results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
#     parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--visualize', action='store_true', help='visualize features')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
#     parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
#     parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
#     parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
#     parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
#     parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
#     opt = parser.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
#     print_args(vars(opt))
#     return opt
#
#
# def main(opt):
#     check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
#     run(**vars(opt))
#
#
# if __name__ == '__main__':
#     opt = parse_opt()
#     main(opt)
# else:
    #sys.path.append('yolov5/')
    # from models import *
    # from utils.utils import *
    # from utils.datasets import *

def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


class MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id=0, num_cls=80):
        super(MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls

    def set_cls_id_attacked(self, cls_id):
        self.cls_id = cls_id

    def forward(self, YOLOoutput):
        batch = YOLOoutput.size()[0]
        output_objectness = YOLOoutput[:, :, 4]
        output_class = YOLOoutput[:, :, 5:(5 + self.num_cls)]
        output_class_new = output_class[:, :, self.cls_id]
        return output_objectness, output_class_new


class DetectorYolov5():
    def __init__(self, cfgfile="yolov5/data/coco128.yaml", weightfile="yolov5/weight/yolov5m.pt",
                 show_detail=False):
        #
        start_t = time.time()

        self.show_detail = show_detail
        # check whether cuda or cpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Set up model

        self.model = DetectMultiBackend(weights=weightfile, data=cfgfile, dnn=False, fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt

        # self.model.eval().cuda()
        self.img_size = 640

        print('Loading Yolov5 weights from %s... Done!' % (weightfile))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.use_cuda = True
            self.model.to(self.device)
            # init MaxProbExtractor
            self.max_prob_extractor = MaxProbExtractor().cuda()
        else:
            self.use_cuda = False
            # init MaxProbExtractor
            self.max_prob_extractor = MaxProbExtractor()

        finish_t = time.time()
        if self.show_detail:
            print('Total init time :%f ' % (finish_t - start_t))

    def detect(self, input_imgs, cls_id_attacked, clear_imgs=None, with_bbox=True):
        start_t = time.time()
        # resize image
        input_imgs = F.interpolate(input_imgs, size=self.img_size).to(self.device)

        # Get detections
        self.model.eval()
        detections = self.model(input_imgs)[0]  ## v5:torch.Size([8, 25200, 85])
        if not (detections[0] == None):
            # init cls_id_attacked
            self.max_prob_extractor.set_cls_id_attacked(cls_id_attacked)
            # max_prob_obj, max_prob_cls = self.max_prob_extractor(detections)
            output_objectness, output_class = self.max_prob_extractor(detections)
            # print(output_objectness.shape, output_class.shape)
            output_cls_obj = torch.mul(output_objectness, output_class)
            max_prob_obj_cls, max_prob_obj_cls_index = torch.max(output_cls_obj, dim=1)
            # print(max_prob_obj_cls.shape)

            if (with_bbox):
                bboxes = non_max_suppression(detections, conf_thres=0.5, iou_thres=0.4, classes=11)  ## <class 'list'>.
                # print(bboxes)
                # bboxes = non_max_suppression_old(detections, 0.4, 0.6)
                # only non None. Replace None with torch.tensor([])
                bboxes = [torch.tensor([]) if bbox is None else bbox for bbox in bboxes]
                bboxes = [rescale_boxes(bbox, self.img_size, [1, 1]) if bbox.dim() == 2 else bbox for bbox in
                          bboxes]  # shape [1,1] means the range of value is [0,1]
                # print("bboxes size : "+str(len(bboxes)))
                # print("bboxes      : "+str(bboxes))

            # get overlap_score
            if not (clear_imgs == None):
                # resize image
                input_imgs_clear = F.interpolate(clear_imgs, size=self.img_size).to(self.device)
                # detections_tensor
                detections_clear = self.model(input_imgs_clear)[0]  ## v5:torch.Size([8, 25200, 85])
                if not (detections_clear[0] == None):
                    #
                    # output_score_clear = self.max_prob_extractor(detections_clear)
                    output_score_obj_clear, output_score_cls_clear = self.max_prob_extractor(detections_clear)
                    output_cls_obj = output_score_obj_clear * output_score_cls_clear
                    # st()
                    output_score_clear, output_score_clear_index = torch.max(output_cls_obj, dim=1)
                    # count overlap
                    output_score = max_prob_obj_cls
                    # output_score_clear = (max_prob_obj_clear * max_prob_cls_clear)
                    overlap_score = torch.abs(output_score - output_score_clear)
                else:
                    overlap_score = torch.tensor(0).to(self.device)
            else:
                overlap_score = torch.FloatTensor(0).to(self.device)
        else:
            print("None : " + str(type(detections)))
            print("None : " + str(detections))
            max_prob_obj = []
            max_prob_cls = []
            bboxes = []
            overlap_score = torch.tensor(0).to(self.device)

        finish_t = time.time()
        if self.show_detail:
            print('Total init time :%f ' % (finish_t - start_t))
        if (with_bbox):
            # return max_prob_obj, max_prob_cls, overlap_score, bboxes
            return max_prob_obj_cls, overlap_score, bboxes
        else:
            # return max_prob_obj, max_prob_cls, overlap_score, [[]]
            return max_prob_obj_cls, overlap_score, [[]]

