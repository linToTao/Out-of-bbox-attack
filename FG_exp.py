# 改 global_dir writer_dir loss
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
# from stylegan2_pytorch import run_generator

### -----------------------------------------------------Initialize Setting     ---------------------------------------------------------------------- ###
Gparser = argparse.ArgumentParser(description='Advpatch Training')
apt = Gparser.parse_known_args()[0]

enable_no_random = True  # ignore EOT (focus on digital space)


Seed = 11111
torch.manual_seed(Seed)
torch.cuda.manual_seed(Seed)
torch.cuda.manual_seed_all(Seed)
np.random.seed(Seed)
random.seed(Seed)
plt2tensor = transforms.Compose([
    transforms.ToTensor()])
device = get_default_device()  # cuda or cpu

use_APGD = True  # True or False
if use_APGD:
    print("use APGD !!!")
use_FG = True  # True or False
if use_FG:
    print("use FG !!!")
if use_APGD:
    num_compare = 3
    queue_len = 20  # 20
    ckp_interval = queue_len * (num_compare + 1)
    epsilon1, epsilon2 = 0.005, 0.008

learning_rate = 4/255  # training learning rate. (hint v3~v4(~0.02) v2(~0.01))

attack_mode = 'trigger'  # 'trigger', 'patch'
position = 'bottom'
if attack_mode == 'patch':
    num_of_patches = 2
if attack_mode == 'trigger':
    num_of_patches = 1

yolo_tiny = False  # hint    : only yolov3 and yolov4
dataset_second = "stop"  # options : inria, stop, test
by_rectangle = False  # True: The patch on the character is "rectangular". / False: The patch on the character is "square"
# transformation options
enable_rotation = False
enable_randomLocation = False
enable_crease = False
enable_projection = False
enable_rectOccluding = False
enable_blurred = False

# output images with bbox
enable_with_bbox = True  # hint    : It is very time consuming. So, the result is only with bbox at checkpoint.
# other setting
enable_show_plt = False  # check output images during training  by human
enable_clear_output = False  # True: training data without any patch
multi_score = True  # True: detection score is "class score * objectness score" for yolo.  /  False: detection score is only "objectness score" for yolo.

# loss weight
weight_loss_tv = 0.5  # total variation loss rate
weight_loss_FG = 0.1  # 0.1 0.5
if use_FG:
    print("weight_loss_FG = " + str(weight_loss_FG))
weight_loss_overlap = 0.0  # total bbox overlap loss rate ([0-0.1])

# training setting
retrain_gan = False  # whether use pre-trained checkpoint

patch_scale = 0.64  # the scale of the patch attached to persons 0.2  (patch_scale, bias_coordinate)=(0.32, 1.5)
bias_coordinate = 1.5 # use in ‘trigger’ attack_mode

n_epochs = 800  # training total epoch
start_epoch = 1  # from what epoch to start training

epoch_save = 800  # from how many A to save a checkpoint
cls_id_attacked = 11  # the class attacked. (0: person   11: stop sign). List: https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda
cls_id_generation = 259  # the class generated at patch. (259: pomeranian) List: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
alpha_latent = 1.0  # weight latent space. z = (alpha_latent * z) + ((1-alpha_latent) * rand_z); std:0.99
rowPatches_size = 128  # the size of patch without gan. It's just like "https://openaccess.thecvf.com/content_CVPRW_2019/html/CV-COPS/Thys_Fooling_Automated_Surveillance_Cameras_Adversarial_Patches_to_Attack_Person_Detection_CVPRW_2019_paper.html"
method_num = 0  # options : 0 (rowPatch without GAN. randon) / 2 (BigGAN) / 3 (styleGAN2)
# parameters of BigGAN
enable_shift_deformator = False  # True: patch = G(deformator(z))  /  False: patch = G(z)
enable_human_annotated_directions = False  # True: only vectors that annotated by human  /   False: all latent vectors
max_value_latent_item = 10  # the max value of latent vectors
# enable_latent_clipping = True    # added by kung. To clip the latent code when optimize
# pre-trained checkpoint
checkpoint_path = "checkpoint/gan_params_10.pt"  # if "retrain_gan" equal "True", and then use this path.
# pre latent vectors
enable_init_latent_vectors = False  # True: patch = G(init_z)  /  False: patch = G(randon_z)
latent_vectors_pathes = ["./exp/exp/generated/generated-vector-5630.pt"]
enable_show_init_patch = False  # check init-patch by human
enable_discriminator = False

if not enable_shift_deformator:
    enable_human_annotated_directions = False
rowPatch_size = 128  # the size of patch without gan. It's just like "https://openaccess.thecvf.com/content_CVPRW_2019/html/CV-COPS/Thys_Fooling_Automated_Surveillance_Cameras_Adversarial_Patches_to_Attack_Person_Detection_CVPRW_2019_paper.html"

label_folder_name = 'yolo-labels_' + str(model_name)
# if model_name == "yolov3" or model_name == "yolov4":
#     if (yolo_tiny):
#         label_folder_name = label_folder_name + 'tiny'

# confirm folder
# global_dir = increment_path(Path('./exp') / 'FG_exp_1', exist_ok=False)  # 'checkpoint'
# global_dir = increment_path(Path('./exp') / 'FG_exp_2', exist_ok=False)  # 'checkpoint'
global_dir = increment_path(Path('./exp') / 'FG_exp_3', exist_ok=False)  # 'checkpoint'
global_dir = Path(global_dir)
checkpoint_dir = global_dir / 'checkpoint'
checkpoint_dir.mkdir(parents=True, exist_ok=True)
sample_dir = global_dir / 'generated'
sample_dir.mkdir(parents=True, exist_ok=True)
print(f"\n##### The results are saved at {global_dir}. #######\n")
np.savetxt(f"./{global_dir}/{apt}--latent:{max_value_latent_item}_normal.txt",
           [enable_rotation, enable_randomLocation, enable_crease, enable_projection, enable_rectOccluding,
            enable_blurred, ])


def show(img):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)


### -----------------------------------------------------Detector---------------------------------------------------------------------- ###
start = time.time()

yolo_tiny = False

if model_name == "yolov3":
    num_fea_per_img = 3
    detectorYolov3 = DetectorYolov3(show_detail=False, tiny=yolo_tiny, use_FG=use_FG)
    detector = detectorYolov3
    batch_size_second = 16
    cls_conf_threshold = 0.
    ds_image_size_second = 416
    # learing_rate = 0.005
    if yolo_tiny == False:
        batch_size_second = 16

if model_name == "yolov5":
    num_fea_per_img = 3
    detectorYolov5 = DetectorYolov5(show_detail=False, use_FG=use_FG, )
    detector = detectorYolov5
    batch_size_second = 16
    cls_conf_threshold = 0.
    ds_image_size_second = 640
    # learing_rate = 0.005
    if yolo_tiny == False:
        batch_size_second = 16

if model_name == "fasterrcnn":
    # just use fasterrcnn directly
    batch_size_second = 8
    # detector = FasterrcnnResnet50()

if model_name == "maskrcnn":
    detector = None  # MaskrcnnResnet50()

finish = time.time()
print('Load detector in %f seconds.' % (finish - start))

# TV
if device == "cuda":
    total_variation = TotalVariation().cuda()
else:
    total_variation = TotalVariation()

source_folder = "./dataset/coco/train_stop_images/"  # "./dataset/coco/test_stop_images/", "./dataset/coco/train_stop_images/"
features_folder = './dataset/coco/train_image_feature-' + model_name
lab_dir = "./dataset/coco/train_stop_labels/"
start_r = time.time()
print("Start to read images from folder")
images = []
features = []
filenames = []
labs = []
for filename in os.listdir(source_folder):
    # print(filename)
    # break
    if (filename.endswith('.jpg') or filename.endswith('.png')):
        # image = imageio.v2.imread(source_folder + filename)
        image = Image.open(source_folder + filename).convert('RGB')
        images.append(image)
        filenames.append(filename[:-4])

        fea = []
        fea_path = os.path.join(features_folder, filename).replace('.jpg', '').replace('.png', '')
        fea_path_0 = fea_path + '-0' + '.pt'
        fea0 = torch.load(fea_path_0).to(device)
        fea.append(fea0)
        fea_path_1 = fea_path + '-1' + '.pt'
        fea1 = torch.load(fea_path_1).to(device)
        fea.append(fea1)
        fea_path_2 = fea_path + '-2' + '.pt'
        fea2 = torch.load(fea_path_2).to(device)
        fea.append(fea2)
        features.append(fea)

        lab_path = os.path.join(lab_dir, filename).replace('.jpg', '.txt').replace('.png', '.txt')
        label = np.loadtxt(lab_path)
        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)
        labs.append(label)
# number of frames
nframes = len(images)
source_data = images
output_name = filenames
finish_r = time.time()
print('Finish reading images in %f seconds.' % (finish_r - start_r))

num_attack_0 = 0
num_attack_1 = 0
num_attack_2 = 0
for i, (imm, feature, lab) in enumerate(zip(source_data, features, labs)):
    print("---------------------------------------------------------")
    print("train on " + filenames[i])
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
            lab[:, [1]] = (lab[:, [1]] * w + padding) / h
            lab[:, [3]] = (lab[:, [3]] * w / h)
        else:
            padding = (w - h) / 2
            padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
            padded_img.paste(img, (0, int(padding)))
            lab[:, [2]] = (lab[:, [2]] * h + padding) / w
            lab[:, [4]] = (lab[:, [4]] * h / w)

    resize = transforms.Resize((ds_image_size_second, ds_image_size_second))
    img = resize(padded_img)  # choose here
    # to tensor
    imm_tensor = plt2tensor(img).unsqueeze(0)
    imm_tensor = imm_tensor.to(device, torch.float)
    img_side = imm_tensor.size()[-1]
    img_output = imm_tensor

    pad_size = 14 - lab.shape[0]
    if (pad_size > 0):
        padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1).unsqueeze(0).to(device)
    else:
        padded_lab = lab.unsqueeze(0).to(device)

    rowPatches = []
    for _ in range(num_of_patches):
        if cls_id_attacked == 11:
            if attack_mode == 'patch':
                rowPatch = torch.rand((3, int(rowPatch_size / 2), rowPatch_size), device=device).requires_grad_(
                    True)  # the delta
            elif attack_mode == 'trigger':
                rowPatch = torch.full((3, int(rowPatch_size / 2), rowPatch_size), 0.5).to(device).requires_grad_(True)
        elif cls_id_attacked == 0:
            rowPatch = torch.rand((3, rowPatch_size, rowPatch_size), device=device).requires_grad_(True)
        rowPatches.append(rowPatch)
    # BigGAN input.     input = ((1-alpha) * fixed) + (alpha * delta)
    fixed_latent_biggan = torch.rand(128, device=device)  # the fixed
    # latent_shift_biggan = torch.rand(len_latent, device=device).requires_grad_(True)
    latent_shift_biggan = torch.normal(0.0, torch.ones(128)).to(device).requires_grad_(True)  # the delta
    # print(latent_shift_biggan)
    enable_init_latent_vectors = False
    if enable_init_latent_vectors:
        rowPatches = []
        for latent_vectors_path in latent_vectors_pathes:
            rowPatch = torch.load(latent_vectors_path).detach()
            rowPatch = rowPatch.to(device).requires_grad_(True)
            rowPatches.append(rowPatch)

    opt_ap = IFGSM([*rowPatches], lr=learning_rate)
    # writer = init_tensorboard_for_FGexp(path=global_dir, name="adversarial" + filenames[i] + "_use_FG")
    # writer = init_tensorboard_for_FGexp(path=global_dir, name="adversarial" + filenames[i] + "_use_det")
    writer = init_tensorboard_for_FGexp(path=global_dir, name="adversarial" + filenames[i] + "_detFG")

    # init & and show the length of one epoch

    if torch.cuda.is_available():
        if attack_mode == 'patch':
            patch_transformer = PatchTransformer().cuda()
        elif attack_mode == 'trigger':
            patch_transformer = PatchTransformer_out_of_bbox(bias_coordinate).cuda()
        patch_applier = PatchApplier().cuda()
    p_img_batch = []
    fake_images_denorm = []

    main_generator = None
    main_scheduler = None
    main_optimizer = None
    main_latentShift = None
    main_denormalisation = None
    main_deformator = None
    if method_num == 0:
        # without GAN, just do gradient-descent with patch
        main_discriminator = None
        # main_scheduler = scheduler_ap
        main_optimizer = opt_ap
        main_latentShift = rowPatch
        main_denormalisation = False

    best_loss = 9999999
    best_epoch = 0
    for epoch in tqdm(range(1, 501), desc='Training for single image'):
        ep_loss_det = 0
        ep_loss_tv = 0
        ep_loss_overlap = 0
        ep_loss_center_bias = 0
        ep_loss_FG = 0

        with autograd.detect_anomaly():
            # only save the patched image, then enable_with_bbox. To reduce time consuming.
            if epoch % epoch_save == 0:
                enable_with_bbox_dynamic = enable_with_bbox
            else:
                enable_with_bbox_dynamic = True  # False

            for rowPatch in rowPatches:
                rowPatch = torch.clamp(rowPatch, 0.000001, 0.99999)
                rowPatch.data = torch.round(rowPatch.data * 10000) * (10 ** -4)
            loss_det, loss_filter_det, loss_overlap, loss_FG, loss_tv, p_img_batch, fake_images_denorm, D_loss = train_rowPtach(
                method_num=method_num, generator=main_generator
                , discriminator=main_discriminator
                , opt=main_optimizer, batch_size=batch_size_second, device=device
                , latent_shift=rowPatches, alpah_latent=alpha_latent, feature=feature
                , input_imgs=imm_tensor, label=padded_lab, patch_scale=patch_scale, cls_id_attacked=cls_id_attacked
                , denormalisation=main_denormalisation
                , model_name=model_name, detector=detector
                , patch_transformer=patch_transformer, patch_applier=patch_applier
                , total_variation=total_variation
                , by_rectangle=by_rectangle
                , enable_rotation=enable_rotation
                , enable_randomLocation=enable_randomLocation
                , enable_crease=enable_crease
                , enable_projection=enable_projection
                , enable_rectOccluding=enable_rectOccluding
                , enable_blurred=enable_blurred
                , enable_with_bbox=enable_with_bbox_dynamic
                , enable_show_plt=enable_show_plt
                , enable_clear_output=enable_clear_output
                , enable_no_random=enable_no_random
                , weight_loss_tv=weight_loss_tv
                , weight_loss_FG=weight_loss_FG
                , weight_loss_overlap=weight_loss_overlap
                , multi_score=multi_score
                , deformator=main_deformator
                , fixed_latent_biggan=fixed_latent_biggan
                , max_value_latent_item=max_value_latent_item
                , enable_shift_deformator=enable_shift_deformator
                , attack_mode=attack_mode
                , position=position
                , img_size=ds_image_size_second
                , use_FG=True)

            # Tloss.backward()
            # opt_ld.step()
            # # Record loss and score
            ep_loss_det += loss_det
            ep_loss_overlap += loss_overlap
            ep_loss_tv += loss_tv
            ep_loss_FG += loss_FG

            # break
        epoch_length_second = 1.
        ep_loss_det = ep_loss_det / epoch_length_second
        ep_loss_overlap = ep_loss_overlap / epoch_length_second
        ep_loss_tv = ep_loss_tv / epoch_length_second
        ep_loss_FG = ep_loss_FG / epoch_length_second

        ep_loss = ep_loss_det


        if ep_loss < best_loss:
            best_loss = ep_loss
            best_epoch = epoch

        ep_loss_det = ep_loss_det.detach().cpu().numpy()
        # ep_loss_overlap = ep_loss_overlap.detach().cpu().numpy()
        ep_loss_overlap = 0
        ep_loss_tv = ep_loss_tv.detach().cpu().numpy()
        # ep_loss_tv = 0
        # ep_loss_FIR = ep_loss_FIR.detach().cpu().numpy()
        ep_loss_FIR = 0

        writer.add_scalar('ep_loss_det', ep_loss_det, epoch)
        # writer.add_scalar('ep_loss_overlap', ep_loss_overlap, epoch)
        # writer.add_scalar('ep_loss_tv', ep_loss_tv, epoch)
        writer.add_scalar('ep_loss_FG', ep_loss_FG, epoch)
        writer.add_scalar('best_epoch', best_epoch, epoch)
        # writer.add_scalar('D_loss', D_loss, epoch)

        # print("epoch                : " + str(epoch))
        # # print("opt_lr               : " + str(main_optimizer.lr * 255))
        # print("ep_loss_det          : " + str(ep_loss_det))
        # print("ep_loss_FG           : " + str(ep_loss_FG))
        # print("best_loss            : " + str(best_loss.data))
        # print("ep_loss_overlap      : " + str(ep_loss_overlap))
        # print("ep_loss_tv           : " + str(ep_loss_tv))
        # print("ep_loss_center_bias  : " + str(ep_loss_center_bias))
        # print("ep_loss_FIR          : " + str(ep_loss_FIR))
        # print("D_loss               : " + str(D_loss))
        # print("latent code:         :'" + f"norn_inf:{torch.max(torch.abs(latent_shift_biggan)):.4f}; norm_1:{torch.norm(latent_shift_biggan, p=1) / latent_shift_biggan.shape[0]:.4f}")
        # max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch,
        #                                                                   cls_id_attacked=cls_id_attacked,
        #                                                                   clear_imgs=None,
        #                                                                   with_bbox=True)
        # print(bboxes)
        # bboxes = []

        if method_num == 0 and (epoch % 50 == 0):
            # save patch
            sample_per_image_dir = sample_dir / filenames[i]
            sample_per_image_dir.mkdir(parents=True, exist_ok=True)
            save_samples(index=epoch, sample_dir=sample_per_image_dir, patches=rowPatches)

            # show(p_img_batch[0].cpu().detach())

    writer.close()
    if best_loss.data <= 0.2:
        num_attack_0 += 1
    if best_loss.data <= 0.3:
        num_attack_1 += 1
    if best_loss.data <= 0.4:
        num_attack_2 += 1
print("@"*15)
print(num_attack_0, num_attack_1, num_attack_2)
print("@"*15)
