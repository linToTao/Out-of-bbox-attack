import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import sys
import random
import argparse

model_name = "yolov5"  # options : yolov3, yolov5, fasterrcnn
if model_name == "yolov3":
    from PyTorchYOLOv3.detect import DetectorYolov3
if model_name == "yolov5":
    from yolov5.detect import DetectorYolov5
import numpy as np
from tqdm import tqdm
from torch import autograd
from torch.utils.data import DataLoader
from ensemble_tool.utils import *
from ensemble_tool.model import train_rowPtach, TotalVariation, IFGSM
# from GANLatentDiscovery.loading import load_from_dir
# from GANLatentDiscovery.utils import is_conditional
# from pytorch_pretrained_detection import FasterrcnnResnet50, MaskrcnnResnet50

from pytorchYOLOv4.demo import DetectorYolov4

from adversarialYolo.demo import DetectorYolov2
from adversarialYolo.train_patch import PatchTrainer
from adversarialYolo.load_data import AdvDataset, PatchTransformer, PatchApplier, PatchTransformer_out_of_bbox
from pathlib import Path
# from stylegan2_pytorch import run_generator

### -----------------------------------------------------Initialize Setting     ---------------------------------------------------------------------- ###
Gparser = argparse.ArgumentParser(description='Advpatch Training')
apt = Gparser.parse_known_args()[0]

enable_no_random = False  # ignore EOT (focus on digital space)


Seed = 11111
torch.manual_seed(Seed)
torch.cuda.manual_seed(Seed)
torch.cuda.manual_seed_all(Seed)
np.random.seed(Seed)
random.seed(Seed)
device = get_default_device()  # cuda or cpu

use_APGD = False  # True or False
learning_rate = 16/255  # training learning rate. (hint v3~v4(~0.02) v2(~0.01))

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
weight_loss_center_bias = 1
weight_loss_FIR = 0.1
weight_loss_overlap = 0.0  # total bbox overlap loss rate ([0-0.1])

# training setting
retrain_gan = False  # whether use pre-trained checkpoint

patch_scale = 0.64  # the scale of the patch attached to persons 0.2  (patch_scale, bias_coordinate)=(0.32, 1.5)
bias_coordinate = 1.5 # use in ‘trigger’ attack_mode

n_epochs = 9999  # training total epoch
start_epoch = 1  # from what epoch to start training

epoch_save = 9999  # from how many A to save a checkpoint
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
global_dir = increment_path(Path('./exp') / 'exp', exist_ok=False)  # 'checkpoint'
global_dir = Path(global_dir)
checkpoint_dir = global_dir / 'checkpoint'
checkpoint_dir.mkdir(parents=True, exist_ok=True)
sample_dir = global_dir / 'generated'
sample_dir.mkdir(parents=True, exist_ok=True)
print(f"\n##### The results are saved at {global_dir}. #######\n")
np.savetxt(f"./{global_dir}/{apt}--latent:{max_value_latent_item}_normal.txt",
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
def show(img):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

### -----------------------------------------------------Detector---------------------------------------------------------------------- ###
start = time.time()

yolo_tiny = False

if model_name == "yolov3":

    detectorYolov3 = DetectorYolov3(show_detail=False, tiny=yolo_tiny)
    detector = detectorYolov3
    batch_size_second = 16
    cls_conf_threshold = 0.
    ds_image_size_second = 416
    # learing_rate = 0.005
    if yolo_tiny == False:
        batch_size_second = 16

if model_name == "yolov5":

    detectorYolov5 = DetectorYolov5(show_detail=False)
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

### -----------------------------------------------------Dataloader---------------------------------------------------------------------- ###
if dataset_second == "inria":
    # InriaDataset

    # batch_size_second      = 8
    train_loader_second = torch.utils.data.DataLoader(AdvDataset(img_dir='./dataset/Inria/Train/pos',
                                                                   lab_dir='./dataset/Inria/Train/pos/' + str(
                                                                       label_folder_name),
                                                                   max_lab=14,
                                                                   imgsize=ds_image_size_second,
                                                                   shuffle=True),
                                                      batch_size=batch_size_second,
                                                      shuffle=True,
                                                      num_workers=10)

if dataset_second == "stop":
    # StopDataset

    # batch_size_second      = 8
    train_loader_second = torch.utils.data.DataLoader(AdvDataset(img_dir='./dataset/coco/train_stop_images',
                                                                   lab_dir='./dataset/coco/train_stop_labels',
                                                                   max_lab=14,
                                                                   imgsize=ds_image_size_second,
                                                                   shuffle=True),
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
# opt_ap = torch.optim.Adam([*rowPatches], lr=learing_rate, betas=(0.5, 0.999), amsgrad=True)
# opt_ld = torch.optim.Adam([latent_shift_biggan], lr=learing_rate, betas=(0.5, 0.999), amsgrad=True)
# opt_ld = torch.optim.SGD([latent_shift_biggan], lr=learing_rate, momentum=0.9)
# optimizer lr_scheduler
# scheduler_ap = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_ap, 'min', patience=50)
# scheduler_ld = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_ld, 'min', patience=50)
# # load checkpoint
# if (retrain_gan):
#     PATH = checkpoint_path
#     #
#     checkpoint = torch.load(PATH)
#     epoch_start = checkpoint['epoch']
#     start_epoch = epoch_start
#     latent_shift_biggan = checkpoint['latent_shift_biggan'].to(device).requires_grad_(True)
#     opt_ld = torch.optim.Adam([latent_shift_biggan], lr=learing_rate, betas=(0.5, 0.999), amsgrad=True)
#     # The reason for DISABLE this is that if we don’t do this, the training results will be very similar.
#     # opt_ld.load_state_dict(checkpoint['optimizer_state_dict_biggan'])

#     # optimizer lr_scheduler
#     scheduler_ld = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_ld, 'min', patience=50)
writer = init_tensorboard(path=global_dir, name="gan_adversarial")

# init & and show the length of one epoch
print(f'One epoch lenght is {len(train_loader_second)}')
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

### -----------------------------------------------------Start training---------------------------------------------------------------------- ###
best_epoch = 0
best_loss = 99999999

for epoch in range(start_epoch, n_epochs + 1):
    ep_loss_det = 0
    ep_loss_filter_det = 0
    ep_loss_tv = 0
    ep_loss_overlap = 0
    ep_loss_center_bias = 0
    ep_loss_FIR = 0
    for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader_second), desc=f'2 Running epoch {epoch}',
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
            loss_det, loss_filter_det, loss_overlap, loss_tv, p_img_batch, fake_images_denorm, D_loss = train_rowPtach(
                method_num=method_num, generator=main_generator
                , discriminator=main_discriminator
                , opt=main_optimizer, batch_size=batch_size_second, device=device
                , latent_shift=rowPatches, alpah_latent=alpha_latent
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
                , weight_loss_FIR=weight_loss_FIR
                , weight_loss_center_bias=weight_loss_center_bias
                , weight_loss_overlap=weight_loss_overlap
                , multi_score=multi_score
                , deformator=main_deformator
                , fixed_latent_biggan=fixed_latent_biggan
                , max_value_latent_item=max_value_latent_item
                , enable_shift_deformator=enable_shift_deformator
                , attack_mode=attack_mode
                , position=position
                , img_size=ds_image_size_second)

            # Tloss.backward()
            # opt_ld.step()
            # # Record loss and score
            ep_loss_det += loss_det
            ep_loss_filter_det += loss_filter_det
            ep_loss_overlap += loss_overlap
            ep_loss_tv += loss_tv

    # if enable_latent_clipping:

    ep_loss_det = ep_loss_det / epoch_length_second
    ep_loss_filter_det = ep_loss_filter_det / epoch_length_second
    ep_loss_overlap = ep_loss_overlap / epoch_length_second
    ep_loss_tv = ep_loss_tv / epoch_length_second

    ep_loss = ep_loss_det  # + (weight_loss_overlap * ep_loss_overlap) + (weight_loss_tv * ep_loss_tv) + (weight_loss_center_bias * ep_loss_center_bias)  # + (weight_loss_FIR * ep_loss_FIR)

    if ep_loss < best_loss:
        best_loss = ep_loss
        best_epoch = epoch

    # main_scheduler.step(ep_loss)

    ep_loss_det = ep_loss_det.detach().cpu().numpy()
    ep_loss_filter_det = ep_loss_filter_det.detach().cpu().numpy()
    # ep_loss_overlap = ep_loss_overlap.detach().cpu().numpy()
    ep_loss_overlap = 0
    ep_loss_tv = ep_loss_tv.detach().cpu().numpy()
    # ep_loss_tv = 0
    # ep_loss_center_bias = ep_loss_center_bias.detach().cpu().numpy()
    ep_loss_center_bias = 0
    # ep_loss_FIR = ep_loss_FIR.detach().cpu().numpy()
    ep_loss_FIR = 0

    writer.add_scalar('ep_loss_det', ep_loss_det, epoch)
    writer.add_scalar('ep_loss_filter_det', ep_loss_filter_det, epoch)
    writer.add_scalar('ep_loss_overlap', ep_loss_overlap, epoch)
    writer.add_scalar('ep_loss_tv', ep_loss_tv, epoch)
    writer.add_scalar('ep_loss_center_bias', ep_loss_center_bias, epoch)
    writer.add_scalar('ep_loss_FIR', ep_loss_FIR, epoch)
    writer.add_scalar('D_loss', D_loss, epoch)

    print("-----------------------------------------------")
    print("epoch                : " + str(epoch))
    print("ep_loss_det          : " + str(ep_loss_det))
    print("ep_loss_filter_fet   : " + str(ep_loss_filter_det))
    print("ep_loss_tv           : " + str(ep_loss_tv))
    print("best_epoch           : " + str(best_epoch))
    print("best_loss          : " + str(best_loss.detach().cpu().numpy()))
    # print("latent code:         :'" + f"norn_inf:{torch.max(torch.abs(latent_shift_biggan)):.4f}; norm_1:{torch.norm(latent_shift_biggan, p=1) / latent_shift_biggan.shape[0]:.4f}")

    if method_num == 0:
        # save patch
        save_samples(index=epoch, sample_dir=sample_dir, patches=rowPatches)
    # if method_num == 2:
    #     # save patch
    #     print(f"Save at: {global_dir}")
    #     save_samples_GANLatentDiscovery(method_num=method_num,
    #                                     index=epoch, sample_dir=sample_dir,
    #                                     deformator=deformator, G=main_generator,
    #                                     latent_shift=latent_shift_biggan, param_rowPatch_latent=alpha_latent,
    #                                     fixed_rand_latent=fixed_latent_biggan,
    #                                     max_value_latent_item=max_value_latent_item,
    #                                     enable_shift_deformator=enable_shift_deformator,
    #                                     device=device)
    #     # print(latent_shift_biggan)
    #     # break
    # elif method_num == 3:
    #     save_samples_GANLatentDiscovery(method_num=method_num,
    #                                     index=epoch, sample_dir=sample_dir,
    #                                     deformator=None, G=main_generator,
    #                                     latent_shift=latent_shift_biggan, param_rowPatch_latent=alpha_latent,
    #                                     fixed_rand_latent=fixed_latent_biggan,
    #                                     max_value_latent_item=max_value_latent_item,
    #                                     enable_shift_deformator=enable_shift_deformator,
    #                                     device=device)

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
