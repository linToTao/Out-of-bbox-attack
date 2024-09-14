import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import sys
import random
import numpy as np
import torch
Seed = 7447  # random.randint(0, 100000)  # 7447  # 3407  # 11111
print(Seed)
torch.manual_seed(Seed)
torch.cuda.manual_seed(Seed)
torch.cuda.manual_seed_all(Seed)
np.random.seed(Seed)
random.seed(Seed)

import argparse

model_name = "yolov5"  # options : yolov3, yolov5
opt_type = 'SGD'
print('opt_type: ' + opt_type)
use_APGD = False  # True or False
use_FG = False  # True or False
is_surrogate = False
# weight_loss_FG = 0.05  # 0.1 0.5 1    0.05 0.1 0.2
learning_rate = 0.1  # 8/255  # training learning rate. 16/255
if model_name == "yolov3":
    if is_surrogate == False:
        from PyTorchYOLOv3.detect import DetectorYolov3
        weight_loss_FG = 0.5  # 0.1 0.5 1
        print("Victim model is yolov3")

if model_name == "yolov5":
    from yolov5.detect import DetectorYolov5
    from yolov5.utils.loss import ComputeLoss
    from yolov5.models.yolo import Model
    import yaml
    weight_loss_FG = 0.05  # 0.05 0.1 0.2
    if is_surrogate == False:
        print("Victim model is yolov5")
    if is_surrogate == True:
        print("Victim model is yolov5_surrogate")


from tqdm import tqdm
from torch import autograd
from torch.utils.data import DataLoader
from ensemble_tool.utils import *
from ensemble_tool.model import train_DPtach, TotalVariation, IFGSM

from adversarialYolo.load_data import AdvDataset, SimDataset, PatchTransformer, PatchApplier, PatchTransformer_out_of_bbox
from pathlib import Path
# from stylegan2_pytorch import run_generator

### -----------------------------------------------------Initialize Setting     ---------------------------------------------------------------------- ###
Gparser = argparse.ArgumentParser(description='Advpatch Training')
apt = Gparser.parse_known_args()[0]

enable_no_random = False  # ignore EOT (focus on digital space)



device = get_default_device()  # cuda or cpu


if use_APGD:
    print("use APGD !!!")

if use_FG:
    print("use FG !!!")
if use_APGD:
    num_compare = 3
    queue_len = 20  # 20
    ckp_interval = queue_len * (num_compare + 1)
    epsilon1, epsilon2 = 0.005, 0.008

dataset_second = "stop"  # options : stop, simulator


print("learning_rate = " + str(int(learning_rate*255)))
attack_mode = 'trigger'  # 'trigger', 'patch'
cls_id_attacked = 11  # (11: stop sign  9: traffic light  46: banana  47: apple). List: https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda
if cls_id_attacked == 11:
    dataset_second = "stop"
    # dataset_second = "simulator"
    position = 'down'        # up down left right
    bias_coordinate = 1.5  # use in ‘trigger’ attack_mode
    print("bias_coordinate = " + str(int(bias_coordinate * 10)))
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
# dataset_second = "simulator"
if attack_mode == 'patch':
    num_of_patches = 2
    patch_scale = 0.36

if attack_mode == 'trigger':
    num_of_patches = 1

print("attack mode is " + attack_mode)
yolo_tiny = False  # hint    : only yolov3 and yolov4

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

if use_FG:
    print("weight_loss_FG = " + str(weight_loss_FG))
weight_loss_overlap = 0.0  # total bbox overlap loss rate ([0-0.1])

# training setting
retrain_gan = False  # whether use pre-trained checkpoint

n_epochs = 1000  # 800  # training total epoch
start_epoch = 1  # from what epoch to start training

epoch_save = 800  # from how many A to save a checkpoint
cls_id_generation = 259  # the class generated at patch. (259: pomeranian) List: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
alpha_latent = 1.0  # weight latent space. z = (alpha_latent * z) + ((1-alpha_latent) * rand_z); std:0.99
rowPatches_size = 196 # 128  # the size of patch without gan. It's just like "https://openaccess.thecvf.com/content_CVPRW_2019/html/CV-COPS/Thys_Fooling_Automated_Surveillance_Cameras_Adversarial_Patches_to_Attack_Person_Detection_CVPRW_2019_paper.html"
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
if dataset_second != "simulator":
    # root = './exp'
    root = '/pub/data/lin'
    # global_dir = increment_path(Path('./exp') / 'exp', exist_ok=False)  # 'checkpoint'
    # global_dir = increment_path(Path('./exp_repeat1') / (model_name+'_'+str(int(learning_rate*255))+'_exp'), exist_ok=False)  # 'checkpoint'
    # global_dir = increment_path(Path('./exp_appendix') / (model_name+'_'+str(int(learning_rate*255))+'_exp'), exist_ok=False)  # 'checkpoint'
    global_dir = increment_path(Path(root) / 'exp' / (model_name+'_Dpatch_exp'), exist_ok=False)
    if is_surrogate:
        global_dir = increment_path(
            Path(root) / (model_name + '_surrogate_' + position + '_' + str(int(bias_coordinate * 10))),
            exist_ok=False)

elif dataset_second == "simulator":
    root = '/pub/data/lin'
    global_dir = increment_path(Path(root) / 'exp' / (model_name+'_Dpatch_exp_sim'), exist_ok=False)

global_dir = Path(global_dir)
checkpoint_dir = global_dir / 'checkpoint'
checkpoint_dir.mkdir(parents=True, exist_ok=True)
sample_dir = global_dir / 'generated'
sample_dir.mkdir(parents=True, exist_ok=True)
print(f"\n##### The results are saved at {global_dir}. #######\n")
# np.savetxt(f"./{global_dir}/{apt}--latent:{max_value_latent_item}_normal.txt",
#            [enable_rotation, enable_randomLocation, enable_crease, enable_projection, enable_rectOccluding,
#             enable_blurred, ])
np.savetxt(f"{global_dir}/{apt}--latent:{max_value_latent_item}_normal.txt",
           [enable_rotation, enable_randomLocation, enable_crease, enable_projection, enable_rectOccluding,
            enable_blurred, ])

### -----------------------------------------------------Prepar rowPatches---------------------------------------------------------------------- ###
# rowPatch.         input = delta
rowPatches = []
for _ in range(num_of_patches):
    if cls_id_attacked == 11:
        if attack_mode == 'patch':
            rowPatch = torch.rand((3, int(rowPatch_size/2), rowPatch_size), device=device).requires_grad_(True)  # the delta
        elif attack_mode == 'trigger':
            # rowPatch = torch.rand((3, int(rowPatch_size), rowPatch_size), device=device).requires_grad_(True)  # the delta
            # # random noise with a reddish tone
            # decease_rate = 0.6
            # rowPatch = torch.rand((3, int(rowPatch_size), rowPatch_size), device=device)
            # rowPatch[1, :, :] *= decease_rate  # decease green channel
            # rowPatch[2, :, :] *= decease_rate  # decease blue channal
            # rowPatch = (rowPatch - rowPatch.min()) / (rowPatch.max() - rowPatch.min())
            # rowPatch = rowPatch.requires_grad_(True)
            rowPatch = torch.full((3, int(rowPatch_size/2), rowPatch_size), 0.5).to(device).requires_grad_(True)
    elif cls_id_attacked == 9:
        rowPatch = torch.full((3, rowPatch_size, rowPatch_size), 0.5).to(device).requires_grad_(True)
        # rowPatch = torch.rand((3, rowPatch_size, rowPatch_size), device=device).requires_grad_(True)
    elif cls_id_attacked == 46:
        rowPatch = torch.full((3, rowPatch_size, rowPatch_size), 0.5).to(device).requires_grad_(True)
    elif cls_id_attacked == 47:
        rowPatch = torch.full((3, rowPatch_size, rowPatch_size), 0.5).to(device).requires_grad_(True)
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
    # batch_size_second = 16
    cls_conf_threshold = 0.
    ds_image_size_second = 416

    # learing_rate = 0.005


if model_name == "yolov5":
    if is_surrogate == False:
        num_fea_per_img = 3
        detectorYolov5 = DetectorYolov5(show_detail=False, use_FG=use_FG)
        detector = detectorYolov5
        batch_size_second = 16
        # batch_size_second = 16
        cls_conf_threshold = 0.
        ds_image_size_second = 640
    if is_surrogate == True:
        num_fea_per_img = 3
        detectorYolov5 = DetectorYolov5(weightfile="yolov5/weight/yolov5_surrogate.pt", show_detail=False, use_FG=use_FG)
        detector = detectorYolov5
        batch_size_second = 32
        # batch_size_second = 16
        cls_conf_threshold = 0.
        ds_image_size_second = 640

    # hyp = 'yolov5/data/hyps/hyp.scratch-low.yaml'
    # if isinstance(hyp, str):
    #     with open(hyp, errors='ignore') as f:
    #         hyp = yaml.safe_load(f)  # load hyps dict
    # weights = 'yolov5/weight/yolov5m.pt'
    # ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    # model = Model(ckpt['model'].yaml, ch=3, nc=80, anchors=hyp.get('anchors')).to(device)  # create
    # model.hyp = hyp
    # detector.model = model
    # detector.compute_loss = ComputeLoss(detector.model)  # init loss class



finish = time.time()
print('Load detector in %f seconds.' % (finish - start))

### -----------------------------------------------------Dataloader---------------------------------------------------------------------- ###
if dataset_second != "simulator":
    # StopDataset
    train_loader_second = torch.utils.data.DataLoader(AdvDataset(img_dir='./dataset/coco/train_'+dataset_second+'_images',
                                                                 lab_dir='./dataset/coco/train_'+dataset_second+'_labels',
                                                                 fea_dir='./dataset/coco/train_'+dataset_second+'_feature-'+model_name,
                                                                 max_lab=14,
                                                                 imgsize=ds_image_size_second,
                                                                 shuffle=True,
                                                                 use_FG=use_FG,
                                                                 # mask_dir='./dataset/coco/train_stop_images_withGrayMask'
                                                                 ),
                                                      batch_size=batch_size_second,
                                                      shuffle=True,
                                                      num_workers=10)



if dataset_second == "simulator":
    # StopDataset
    train_loader_second = torch.utils.data.DataLoader(SimDataset(img_dir='./test_images/train_images',
                                                                 lab_dir='./test_images/train_images_yolo-labels_'+model_name,
                                                                 fea_dir='./test_images/train_image_feature-'+model_name,
                                                                 max_lab=14,
                                                                 imgsize=ds_image_size_second,
                                                                 shuffle=True,
                                                                 use_FG=use_FG,
                                                                 # mask_dir='./dataset/coco/train_stop_images_withGrayMask'
                                                                 ),
                                                      batch_size=batch_size_second,
                                                      shuffle=True,
                                                      num_workers=10)

# init
train_loader_second = DeviceDataLoader(train_loader_second, device)

# TV
if device == "cuda":
    total_variation = TotalVariation().cuda()
else:
    total_variation = TotalVariation()

### -----------------------------------------------------Initialize before training---------------------------------------------------------------------- ###
# rowPatch = rowPatch.cuda()
# rowPatch.requires_grad = True
epoch_length_second = len(train_loader_second)
ep_loss_det = 0
ep_loss_tv = 0
torch.cuda.empty_cache()
# Create optimizers

opt_ap = IFGSM([*rowPatches], lr=learning_rate)
opt_adam = torch.optim.Adam([*rowPatches], lr=learning_rate, betas=(0.5, 0.999), amsgrad=True)

writer = init_tensorboard(path=global_dir, name="gan_adversarial")

# init & and show the length of one epoch
print(f'One epoch lenght is {len(train_loader_second)}')
if torch.cuda.is_available():
    if attack_mode == 'patch':
        patch_transformer = PatchTransformer().cuda()
    elif attack_mode == 'trigger':
        patch_transformer = PatchTransformer_out_of_bbox(bias_coordinate, position).cuda()
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
    if opt_type == 'SGD':

        main_optimizer = opt_ap
    elif opt_type == 'adam':
        main_optimizer = opt_adam
    main_latentShift = rowPatch
    main_denormalisation = False

### -----------------------------------------------------Start training---------------------------------------------------------------------- ###
current_lr = learning_rate
best_epoch = 0
best_loss = 9999999

for epoch in range(start_epoch, n_epochs + 1):
    ep_loss_det = 0
    ep_loss_filter_det = 0
    ep_loss_tv = 0
    ep_loss_overlap = 0
    ep_loss_center_bias = 0
    ep_loss_FG = 0
    for i_batch, (img_batch, lab_batch, fea0_batch, fea1_batch, fea2_batch) in tqdm(enumerate(train_loader_second), desc=f'2 Running epoch {epoch}',
                                                total=epoch_length_second):  ## , input_imgs=img_batch, label=lab_batch,
        with autograd.detect_anomaly():
            # only save the patched image, then enable_with_bbox. To reduce time consuming.
            if epoch % epoch_save == 0:
                enable_with_bbox_dynamic = enable_with_bbox
            else:
                enable_with_bbox_dynamic = False  # False

            # Train with GANLatentDiscovery
            # st()
            # opt_ld.zero_grad()
            # np.save('gg', latent_shift_biggan.cpu().detach().numpy())
            # np.argwhere(np.load('gg.npy')!=latent_shift_biggan.cpu().detach().numpy())
            for rowPatch in rowPatches:
                rowPatch.data = torch.round(rowPatch.data * 10000) * (10 ** -4)
            loss, p_img_batch, fake_images_denorm = train_DPtach(
                method_num=method_num
                , opt=main_optimizer, batch_size=batch_size_second, device=device
                , latent_shift=rowPatches, alpah_latent=alpha_latent, feature=[fea0_batch, fea1_batch, fea2_batch]
                , input_imgs=img_batch, label=lab_batch, patch_scale=patch_scale, cls_id_attacked=cls_id_attacked
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
                , img_size=ds_image_size_second
                , use_FG=use_FG
                )


            # Tloss.backward()
            # opt_ld.step()
            # # Record loss and score
            ep_loss_det += loss

    # if enable_latent_clipping:

    ep_loss_det = ep_loss_det / epoch_length_second
    ep_loss_filter_det = ep_loss_filter_det / epoch_length_second
    ep_loss_overlap = ep_loss_overlap / epoch_length_second
    ep_loss_tv = ep_loss_tv / epoch_length_second
    ep_loss_FG = ep_loss_FG / epoch_length_second

    ep_loss = ep_loss_det  # + (weight_loss_overlap * ep_loss_overlap) + (weight_loss_tv * ep_loss_tv) + (weight_loss_center_bias * ep_loss_center_bias)  # + (weight_loss_FIR * ep_loss_FIR)
    # main_scheduler.step(ep_loss)

    ep_loss_det = ep_loss_det.detach().cpu().numpy()

    # ep_loss_FIR = 0

    if ep_loss < best_loss:
        best_loss = ep_loss
        best_epoch = epoch

    writer.add_scalar('ep_loss_det', ep_loss_det, epoch)
    writer.add_scalar('seed', Seed, epoch)

    writer.add_scalar('current_lr', current_lr, epoch)
    writer.add_scalar('ep_loss_overlap', ep_loss_overlap, epoch)
    writer.add_scalar('ep_loss_tv', ep_loss_tv, epoch)
    # writer.add_scalar('ep_loss_center_bias', ep_loss_center_bias, epoch)
    writer.add_scalar('ep_loss_FG', ep_loss_FG, epoch)
    writer.add_scalar('best_epoch', best_epoch, epoch)

    print("-----------------------------------------------")
    print("epoch                : " + str(epoch))
    print("seed                 : " + str(Seed))
    print("ep_loss_det          : " + str(ep_loss_det))
    print("current_lr           : " + str(current_lr * 255))
    print("ep_loss_tv           : " + str(ep_loss_tv))
    print("best_epoch           : " + str(best_epoch))
    # print("best_loss            : " + str(best_loss.detach().cpu().numpy()))
    # print(rowPatches)
    print("-----------------------------------------------")

    if epoch % 5 == 0:
        current_lr = current_lr * 0.95
        main_optimizer = IFGSM([*rowPatches], lr=current_lr)

    if method_num == 0:
        if best_epoch == epoch:
            # for id, patch in enumerate(rowPatches):
            #     if (patch > 1).any() or (patch < 0).any():
            #         print('XXXX')
            save_samples(index=epoch, sample_dir=sample_dir, patches=rowPatches)

    if epoch % epoch_save == 0:
        # # save the patched image
        # print(f"@{global_dir}")
        save_the_patched(index=epoch, the_patched=p_img_batch, sample_dir=sample_dir, show=False)
        # # save checkpoint
        # Additional information
        EPOCH = epoch
        PATH = str(checkpoint_dir) + "/gan_params_" + str(epoch) + ".pt"
        torch.save({
            'epoch': EPOCH,
            'optimizer_state_dict_biggan': opt_ap.state_dict(),
            'latent_shift_biggans': [rowPatch.data for rowPatch in rowPatches],
            'alpha_latent': alpha_latent,  # 'annotated_idx': annotated_idx,
            'enable_shift_deformator': enable_shift_deformator,
            'enable_human_annotated_directions': enable_human_annotated_directions,
            'ep_loss_det': ep_loss_det,
            'ep_loss_overlap': ep_loss_overlap,
            'ep_loss_tv': ep_loss_tv
        }, PATH)
        print(f"save checkpoint: " + str(PATH))
writer.close()
