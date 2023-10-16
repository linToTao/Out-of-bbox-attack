import torch
import torch.nn as nn
import torch.nn.functional as F
from ensemble_tool.utils import denorm
from pytorchYOLOv4.tool.utils import load_class_names
from adversarialYolo.utils import do_detect as get_bbox_yolov2
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import time
import sys
import numpy as np

from GANLatentDiscovery.torch_tools.visualization import to_image
from GANLatentDiscovery.visualization import interpolate, interpolate_shift
from ipdb import set_trace as st


class IFGSM(torch.optim.Optimizer):

    def __init__(self, params, lr):
        self.lr = lr
        super(IFGSM, self).__init__(params, {})

    def step(self, closure=False):
        for param_group in self.param_groups:
            params = param_group['params']
            for param in params:
                param.data = param.data - self.lr * torch.sign(param.grad)


class APGD(torch.optim.Optimizer):

    def __init__(self, params, lr):
        self.lr = lr
        super(IFGSM, self).__init__(params, {})

    def step(self, closure=False):
        for param_group in self.param_groups:
            params = param_group['params']
            for param in params:
                param.data = param.data - self.lr * torch.sign(param.grad)


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)


def eval_rowPtach(generator, batch_size, device
                  , latent_shift, alpah_latent
                  , input_imgs, label, patch_scale, cls_id_attacked
                  , denormalisation
                  , model_name, detector
                  , patch_transformer, patch_applier
                  , by_rectangle
                  , enable_rotation
                  , enable_randomLocation
                  , enable_crease
                  , enable_projection
                  , enable_rectOccluding
                  , enable_blurred
                  , enable_with_bbox
                  , enable_show_plt
                  , enable_clear_output
                  , cls_conf_threshold
                  , patch_mode=0
                  , multi_score=False
                  , enable_no_random=False
                  , fake_images_default=None
                  , attack_mode='trigger'
                  , position='bottom'
                  , img_size=416
                  , no_detect_img=False
                  , low_conf_img=False):

    fake_images = fake_images_default

    # enable_empty_patch == False: no patch
    enable_empty_patch = False
    if (patch_mode == 1):
        enable_empty_patch = True

    min_loss_det = 999
    for i_fimage in range(1):
        p_img_batch = input_imgs[0]
        for id, patch in enumerate(fake_images):
            if not (enable_clear_output):
                if attack_mode == 'patch':
                    adv_batch_t, adv_patch_set, msk_batch = patch_transformer(adv_patch=fake_images[id][i_fimage],
                                                                              lab_batch=label,
                                                                              img_size=img_size,
                                                                              patch_mask=[],
                                                                              cls_id_attacked=cls_id_attacked,
                                                                              by_rectangle=by_rectangle,
                                                                              do_rotate=enable_rotation,
                                                                              rand_loc=enable_randomLocation,
                                                                              with_black_trans=False,
                                                                              scale_rate=patch_scale,
                                                                              with_crease=enable_crease,
                                                                              with_projection=enable_projection,
                                                                              with_rectOccluding=enable_rectOccluding,
                                                                              enable_empty_patch=enable_empty_patch,
                                                                              enable_no_random=enable_no_random,
                                                                              enable_blurred=enable_blurred,
                                                                              num_of_round=id)
                elif attack_mode == 'trigger':
                    adv_batch_t, adv_patch_set, msk_batch = patch_transformer(adv_patch=fake_images[id][i_fimage],
                                                                              lab_batch=label,
                                                                              img_size=img_size,
                                                                              patch_mask=[],
                                                                              cls_id_attacked=cls_id_attacked,
                                                                              by_rectangle=by_rectangle,
                                                                              do_rotate=enable_rotation,
                                                                              rand_loc=enable_randomLocation,
                                                                              with_black_trans=False,
                                                                              scale_rate=patch_scale,
                                                                              with_crease=enable_crease,
                                                                              with_projection=enable_projection,
                                                                              with_rectOccluding=enable_rectOccluding,
                                                                              enable_empty_patch=enable_empty_patch,
                                                                              enable_no_random=enable_no_random,
                                                                              enable_blurred=enable_blurred,
                                                                              position=position)

                p_img_batch = patch_applier(p_img_batch, adv_batch_t)
            else:
                p_img_batch = input_imgs

        # # test
        # print("label       size : "+str(label.size()))
        # print("label            : "+str(label))
        # print("fake_images size : "+str(fake_images.size()))
        # print("p_img_batch size : "+str(p_img_batch.size()))
        # # test
        # import numpy as np
        # def show(img):
        #     npimg = img.numpy()
        #     fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), 
        #                                     interpolation='nearest')
        #     fig.axes.get_xaxis().set_visible(False)
        #     fig.axes.get_yaxis().set_visible(False)
        # plt.ion()
        # plt.figure()
        # show(make_grid(fake_images.detach().cpu()[:16,:,:,:]))
        # plt.figure()
        # show(make_grid(p_img_batch.detach().cpu()[:16,:,:,:]))
        # plt.ioff()
        # plt.show()
        # sys.exit()

        # loss.

        if (model_name == "yolov3"):
            max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch,
                                                                      cls_id_attacked=cls_id_attacked,
                                                                      clear_imgs=input_imgs, with_bbox=enable_with_bbox)
            # loss_det
            if multi_score:
                # multi = max_prob_obj * max_prob_cls
                loss_det = torch.mean(max_prob_obj_cls)
            else:
                loss_det = torch.mean(max_prob_obj_cls)
            # loss_overlap
            loss_overlap = -torch.mean(overlap_score)
        if (model_name == "yolov5"):
            max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch,
                                                                      cls_id_attacked=cls_id_attacked,
                                                                      clear_imgs=input_imgs, with_bbox=enable_with_bbox)
            # loss_det
            if multi_score:
                # multi = max_prob_obj * max_prob_cls
                loss_det = torch.mean(max_prob_obj_cls)
            else:
                loss_det = torch.mean(max_prob_obj_cls)
            # loss_overlap
            loss_overlap = -torch.mean(overlap_score)
        if (model_name == "fasterrcnn"):
            max_prob, bboxes = detector.detect(tensor_image_inputs=p_img_batch, cls_id_attacked=cls_id_attacked,
                                               threshold=0.5)
            # loss_det
            loss_det = torch.mean(max_prob)
            # no loss_overlap
            loss_overlap = torch.tensor(0.0).to(device)

        if (min_loss_det > loss_det):
            min_loss_det = loss_det

    loss_det = min_loss_det

    # darw bbox
    if len(bboxes) == 0:
        no_detect_img = True
    if enable_with_bbox and len(bboxes) > 0:
        trans_2pilimage = transforms.ToPILImage()
        batch = p_img_batch.size()[0]
        for b in range(batch):
            img_pil = trans_2pilimage(p_img_batch[b].cpu())
            img_width = img_pil.size[0]
            img_height = img_pil.size[1]
            namesfile = 'pytorchYOLOv4/data/coco.names'
            class_names = load_class_names(namesfile)
            # sample first image
            # print("bbox : "+str(bbox))
            bbox = bboxes[b]
            for box in bbox:
                # print("box size : "+str(box.size()))
                cls_id = box[6].int()
                cls_name = class_names[cls_id]
                cls_conf = box[5]
                if cls_id == cls_id_attacked:
                    if cls_conf > cls_conf_threshold:
                        low_conf_img = False
                        if model_name == "yolov2":
                            x_center = box[0]
                            y_center = box[1]
                            width = box[2]
                            height = box[3]
                            left = (x_center.item() - width.item() / 2) * img_width
                            right = (x_center.item() + width.item() / 2) * img_width
                            top = (y_center.item() - height.item() / 2) * img_height
                            bottom = (y_center.item() + height.item() / 2) * img_height
                        if (model_name == "yolov5") or (model_name == "yolov3") or (model_name == "fasterrcnn"):
                            left = int(box[0] * img_width)
                            right = int(box[2] * img_width)
                            top = int(box[1] * img_height)
                            bottom = int(box[3] * img_height)
                        # img with prediction
                        draw = ImageDraw.Draw(img_pil)
                        shape = [left, top, right, bottom]
                        draw.rectangle(shape, outline="red")
                        # text
                        color = [255, 0, 0]
                        font = ImageFont.truetype("cmb10.ttf", int(min(img_width, img_height) / 18))
                        sentence = str(cls_name) + " (" + str(round(float(cls_conf), 2)) + ")"
                        position = [left, top]
                        draw.text(tuple(position), sentence, tuple(color), font=font)
                    else:
                        low_conf_img = True
            if enable_show_plt:
                # show, debug
                plt.imshow(img_pil)
                plt.show()
            trans_2tensor = transforms.ToTensor()
            p_img_batch[b] = trans_2tensor(img_pil)

    return p_img_batch, fake_images, bboxes, no_detect_img, low_conf_img


def train_rowPtach(method_num, generator
                   , discriminator
                   , opt, batch_size, device
                   , latent_shift, alpah_latent, feature
                   , input_imgs, label, patch_scale, cls_id_attacked
                   , denormalisation
                   , model_name, detector
                   , patch_transformer, patch_applier
                   , total_variation
                   , by_rectangle
                   , enable_rotation
                   , enable_randomLocation
                   , enable_crease
                   , enable_projection
                   , enable_rectOccluding
                   , enable_blurred
                   , enable_with_bbox
                   , enable_show_plt
                   , enable_clear_output
                   , enable_no_random
                   , weight_loss_tv
                   , weight_loss_FG
                   , weight_loss_overlap
                   , multi_score=False
                   , deformator=None
                   , fixed_latent_biggan=None
                   , max_value_latent_item=8
                   , enable_shift_deformator=True
                   , attack_mode='trigger'
                   , position='bottom'
                   , cls_conf_threshold=0.5
                   , img_size=416
                   , use_FG=False
                   ):
    # Clear generator gradients
    if method_num == 0:
        opt.zero_grad()
        p_img_batch = input_imgs
        for id, patch in enumerate(latent_shift):
            if attack_mode == 'patch':
                adv_batch_t, adv_patch_set, msk_batch = patch_transformer(adv_patch=patch,
                                                                          lab_batch=label,
                                                                          img_size=img_size,
                                                                          patch_mask=[],
                                                                          cls_id_attacked=cls_id_attacked,
                                                                          by_rectangle=by_rectangle,
                                                                          do_rotate=enable_rotation,
                                                                          rand_loc=enable_randomLocation,
                                                                          with_black_trans=False,
                                                                          scale_rate=patch_scale,
                                                                          with_crease=enable_crease,
                                                                          with_projection=enable_projection,
                                                                          with_rectOccluding=enable_rectOccluding,
                                                                          enable_blurred=enable_blurred,
                                                                          enable_no_random=enable_no_random,
                                                                          num_of_round=id)
            elif attack_mode == 'trigger':
                adv_batch_t, adv_patch_set, msk_batch = patch_transformer(adv_patch=patch,
                                                                          lab_batch=label,
                                                                          img_size=img_size,
                                                                          patch_mask=[],
                                                                          cls_id_attacked=cls_id_attacked,
                                                                          by_rectangle=by_rectangle,
                                                                          do_rotate=enable_rotation,
                                                                          rand_loc=enable_randomLocation,
                                                                          with_black_trans=False,
                                                                          scale_rate=patch_scale,
                                                                          with_crease=enable_crease,
                                                                          with_projection=enable_projection,
                                                                          with_rectOccluding=enable_rectOccluding,
                                                                          enable_blurred=enable_blurred,
                                                                          enable_no_random=enable_no_random,
                                                                          position=position)
            p_img_batch = patch_applier(p_img_batch, adv_batch_t)  # torch.Size([8, 14, 3, 416, 416])

        if (model_name == "yolov3"):
            if use_FG:
                max_prob_obj_cls, overlap_score, bboxes, feature_hook = detector.detect(input_imgs=p_img_batch,
                                                                          cls_id_attacked=cls_id_attacked,
                                                                          clear_imgs=None,
                                                                          # clear_imgs=input_imgs,
                                                                          with_bbox=enable_with_bbox)
            else:
                max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch,
                                                                          cls_id_attacked=cls_id_attacked,
                                                                          clear_imgs=None,
                                                                          # clear_imgs=input_imgs,
                                                                          with_bbox=enable_with_bbox)
            # max_prob_obj_cls_clear, overlap_score_clear, bboxes_clear = detector.detect(input_imgs=input_imgs,
            #                                                                       cls_id_attacked=cls_id_attacked,
            #                                                                       clear_imgs=None,
            #                                                                       with_bbox=enable_with_bbox)

            # loss_det
            if multi_score:
                # multi = max_prob_obj * max_prob_cls
                loss_det = torch.mean(max_prob_obj_cls)
                weight = (max_prob_obj_cls / torch.sum(max_prob_obj_cls)).detach()
                loss_filter_det = loss_det
                # loss_filter_det = torch.matmul(weight, max_prob_obj_cls)
                # print("exp3")
                # loss_filter_det = torch.mean(max_prob_obj_cls * max_prob_obj_cls)
                # print("exp2")
                # loss_filter_det = torch.mean(max_prob_obj_cls[max_prob_obj_cls > cls_conf_threshold])
            else:
                loss_det = torch.mean(max_prob_obj_cls)
            # loss_overlap
            loss_overlap = -torch.mean(overlap_score)

            # FG
            if use_FG:
                loss_FG0 = torch.nn.functional.mse_loss(feature[0], feature_hook[0])
                loss_FG1 = torch.nn.functional.mse_loss(feature[1], feature_hook[1])
                loss_FG2 = torch.nn.functional.mse_loss(feature[2], feature_hook[2])
                loss_FG = loss_FG0 + loss_FG1 + loss_FG2
                # print(loss_FG0)
                # print(loss_FG1)
                # print(loss_FG2)
                # print(loss_FG)
                # print(feature[2].size(), feature_hook[2].size())
            else:
                loss_FG = torch.Tensor([0]).to(device)

        if (model_name == "yolov5"):
            if use_FG:
                max_prob_obj_cls, overlap_score, bboxes, feature_hook = detector.detect(input_imgs=p_img_batch,
                                                                          cls_id_attacked=cls_id_attacked,
                                                                          clear_imgs=None,
                                                                          # clear_imgs=input_imgs,
                                                                          with_bbox=enable_with_bbox)
            else:
                max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch,
                                                                          cls_id_attacked=cls_id_attacked,
                                                                          clear_imgs=None,
                                                                          # clear_imgs=input_imgs,
                                                                          with_bbox=enable_with_bbox)
            # max_prob_obj_cls_clear, overlap_score_clear, bboxes_clear = detector.detect(input_imgs=input_imgs,
            #                                                                       cls_id_attacked=cls_id_attacked,
            #                                                                       clear_imgs=None,
            #                                                                       with_bbox=enable_with_bbox)

            # loss_det
            if multi_score:
                # multi = max_prob_obj * max_prob_cls
                loss_det = torch.mean(max_prob_obj_cls)
                weight = (max_prob_obj_cls / torch.sum(max_prob_obj_cls)).detach()
                loss_filter_det = loss_det
                # loss_filter_det = torch.matmul(weight, max_prob_obj_cls)
                # print("exp3")
                # loss_filter_det = torch.mean(max_prob_obj_cls * max_prob_obj_cls)
                # print("exp2")
                # loss_filter_det = torch.mean(max_prob_obj_cls[max_prob_obj_cls > cls_conf_threshold])
            else:
                loss_det = torch.mean(max_prob_obj_cls)
            # loss_overlap
            loss_overlap = -torch.mean(overlap_score)

            # FG

            loss_FG = torch.Tensor([0]).to(device)

        min_loss_det = 999
        if (min_loss_det > loss_filter_det):
            min_loss_det = loss_filter_det

        # loss: total variation
        loss_tv = 0
        for patch in latent_shift:
            loss_tv += total_variation(patch)

        # st()
        # loss
        D_loss = 0

        if discriminator is not None:
            loss = min_loss_det + (weight_loss_tv * loss_tv) + 1e-3 * D_loss
            # loss = min_loss_det + (weight_loss_overlap * loss_overlap) + (weight_loss_tv * loss_tv) + 1e-3 * D_loss
        else:
            loss_overlap = 0
            loss = min_loss_det + (weight_loss_FG * loss_FG) + (weight_loss_tv * loss_tv)  # (weight_loss_center_bias * loss_center_bias)  + (weight_loss_overlap * loss_overlap)
            # loss = loss_FG

        loss.backward()
        # print(latent_shift[0].grad)
        opt.step()

        # draw bbox at output
        if enable_with_bbox and len(bboxes) > 0:
            trans_2pilimage = transforms.ToPILImage()
            batch = p_img_batch.size()[0]
            for b in range(batch):
                img_pil = trans_2pilimage(p_img_batch[b].cpu())
                img_width = img_pil.size[0]
                img_height = img_pil.size[1]
                namesfile = 'pytorchYOLOv4/data/coco.names'
                class_names = load_class_names(namesfile)
                # sample first image
                bbox = bboxes[b]
                # print("bbox : "+str(bbox))
                for box in bbox:
                    # print("box size : "+str(box.size()))
                    cls_id = box[6].int()
                    cls_name = class_names[cls_id]
                    cls_conf = box[5]
                    if (cls_id == cls_id_attacked):
                        if (model_name == "yolov2"):
                            x_center = box[0]
                            y_center = box[1]
                            width = box[2]
                            height = box[3]
                            left = (x_center.item() - width.item() / 2) * img_width
                            right = (x_center.item() + width.item() / 2) * img_width
                            top = (y_center.item() - height.item() / 2) * img_height
                            bottom = (y_center.item() + height.item() / 2) * img_height
                        if (model_name == "yolov5") or (model_name == "yolov3" or (model_name == "fasterrcnn")):
                            left = int(box[0] * img_width)
                            right = int(box[2] * img_width)
                            top = int(box[1] * img_height)
                            bottom = int(box[3] * img_height)
                        # img with prediction
                        draw = ImageDraw.Draw(img_pil)
                        shape = [left, top, right, bottom]
                        draw.rectangle(shape, outline="red")
                        # text
                        color = [255, 0, 0]
                        font = ImageFont.truetype("cmb10.ttf", int(min(img_width, img_height) / 18))
                        sentence = str(cls_name) + " (" + str(round(float(cls_conf), 2)) + ")"
                        position = [left, top]
                        draw.text(tuple(position), sentence, tuple(color), font=font)
                if enable_show_plt:
                    # show, debug
                    plt.imshow(img_pil)
                    plt.show()
                trans_2tensor = transforms.ToTensor()
                p_img_batch[b] = trans_2tensor(img_pil)
        # print("loss_det:",loss)
        # st()
        return loss_det, loss_filter_det, loss_overlap, loss_FG, loss_tv, p_img_batch, latent_shift, D_loss

    # if method_num == 2:  # biggan
    #     opt.zero_grad()
    #     latent_shift.data = torch.clamp(latent_shift.data, -max_value_latent_item, max_value_latent_item)
    #     # Generate fake images
    #     # st()
    #     if not (deformator == None):
    #         # BigGAN
    #         rows = 1
    #         # set desired class for conditional GAN
    #         zs = torch.rand([rows, generator.dim_z] if type(generator.dim_z) == int else [rows] + generator.dim_z,
    #                         device=device)
    #         # Do img = G(z + shift)
    #         z = zs[0]
    #         # print("z:",z[:10])
    #         if enable_shift_deformator:
    #             z = ((1 - alpah_latent) * z) + (alpah_latent * fixed_latent_biggan)
    #             latent_shift = torch.clamp((max_value_latent_item * latent_shift), max=max_value_latent_item,
    #                                        min=-max_value_latent_item)
    #             interpolation_deformed = interpolate_shift(generator, z.unsqueeze(0),
    #                                                        latent_shift=latent_shift.unsqueeze(0),
    #                                                        deformator=deformator,
    #                                                        device=device)
    #         else:
    #             # Directly, change z.  z = z + latent_shift
    #             z = ((1 - alpah_latent) * z) + (alpah_latent * latent_shift)
    #             interpolation_deformed = interpolate_shift(generator, z.unsqueeze(0),
    #                                                        latent_shift=torch.zeros_like(z.unsqueeze(0)).to(device),
    #                                                        deformator=None,
    #                                                        device=device)
    #         interpolation_deformed = interpolation_deformed.unsqueeze(0).to(device)  # torch.Size([1, 3, 128, 128])
    #         # print("dore",interpolation_deformed[0,0,:10,0])
    #         fake_images = (interpolation_deformed + 1) * 0.5
    #         # print(fake_images)
    #         # st()
    #     else:
    #         # Generate fake images
    #         latent = torch.rand(latent_shift.size(), device=device)
    #         latent_merged = ((1 - alpah_latent) * latent) + (alpah_latent * latent_shift)
    #         if not (generator == None):
    #             fake_images = generator(latent_merged)
    #             if (denormalisation):
    #                 fake_images = denorm(fake_images)  # torch.Size([batch, 3, 128, 128]) . batch=1
    #         else:
    #             fake_images = latent_merged.unsqueeze(0)
    #
    #     min_loss_det = 999
    #     for i_fimage in range(1):
    #         if not (enable_clear_output):
    #             adv_batch_t, adv_patch_set, msk_batch = patch_transformer(adv_patch=fake_images[i_fimage],
    #                                                                       lab_batch=label,
    #                                                                       img_size=416,
    #                                                                       patch_mask=[],
    #                                                                       by_rectangle=by_rectangle,
    #                                                                       do_rotate=enable_rotation,
    #                                                                       rand_loc=enable_randomLocation,
    #                                                                       with_black_trans=False,
    #                                                                       scale_rate=patch_scale,
    #                                                                       with_crease=enable_crease,
    #                                                                       with_projection=enable_projection,
    #                                                                       with_rectOccluding=enable_rectOccluding,
    #                                                                       enable_blurred=enable_blurred)
    #             p_img_batch = patch_applier(input_imgs, adv_batch_t)  # torch.Size([8, 14, 3, 416, 416])
    #             # st()
    #
    #             # label       size : torch.Size([16, 14, 5])
    #             # input_imgs  size : torch.Size([16, 3, 416, 416])
    #             # adv_batch_t size : torch.Size([16, 14, 3, 416, 416])
    #             # p_img_batch size : torch.Size([16, 3, 416, 416])
    #             # adv_patch_set size :torch.Size([3, 300, 300])
    #             # print("p_img_batch:",p_img_batch[0,0,:10,0])
    #         else:
    #             p_img_batch = input_imgs
    #         if discriminator is not None:
    #             D_loss = discriminator(fake_images, torch.Tensor([259]).long().cuda())
    #         else:
    #             D_loss = 0
    #         # st()
    #         # loss.
    #         if (model_name == "yolov5"):
    #             max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch,
    #                                                                       cls_id_attacked=cls_id_attacked,
    #                                                                       clear_imgs=input_imgs,
    #                                                                       with_bbox=enable_with_bbox)
    #             # st()
    #             # np.save('gg', p_img_batch.cpu().detach().numpy())
    #             #  gg=np.load('gg.npy')    gg2=p_img_batch.cpu().detach().numpy()  np.argwhere(gg!=gg2)
    #             # loss_det
    #             # default is multi_score. Here, "max_prob_cls" is "cls*obj"
    #             # print(max_prob_obj_cls)
    #             # st()
    #             loss_det = torch.mean(max_prob_obj_cls)
    #             # loss_overlap
    #             loss_overlap = -torch.mean(overlap_score)
    #         if (model_name == "yolov3"):
    #             max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch,
    #                                                                       cls_id_attacked=cls_id_attacked,
    #                                                                       clear_imgs=input_imgs,
    #                                                                       with_bbox=enable_with_bbox)
    #             # loss_det
    #             if multi_score:
    #                 # multi = max_prob_obj * max_prob_cls
    #                 loss_det = torch.mean(max_prob_obj_cls)
    #             else:
    #                 loss_det = torch.mean(max_prob_obj_cls)
    #             # loss_overlap
    #             loss_overlap = -torch.mean(overlap_score)
    #         if (model_name == "yolov2"):
    #             max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch,
    #                                                                       cls_id_attacked=cls_id_attacked,
    #                                                                       clear_imgs=input_imgs,
    #                                                                       with_bbox=enable_with_bbox)
    #             # loss_det
    #             if multi_score:
    #                 loss_det = torch.mean(max_prob_obj_cls)
    #             else:
    #                 loss_det = torch.mean(max_prob_obj_cls)
    #             # loss_overlap
    #             loss_overlap = -torch.mean(overlap_score)
    #         if (model_name == "fasterrcnn"):
    #             max_prob, bboxes = detector.detect(tensor_image_inputs=p_img_batch, cls_id_attacked=cls_id_attacked,
    #                                                threshold=0.5)
    #             # loss_det
    #             loss_det = torch.mean(max_prob)
    #             # no loss_overlap
    #             loss_overlap = torch.tensor(0.0).to(device)
    #
    #         if (min_loss_det > loss_det):
    #             min_loss_det = loss_det
    #
    #     # loss: total variation
    #     loss_tv = total_variation(fake_images[0])
    #     # st()
    #     # loss
    #     if discriminator is not None:
    #         loss = min_loss_det + (weight_loss_overlap * loss_overlap) + (weight_loss_tv * loss_tv) + 1e-3 * D_loss
    #     else:
    #         loss = min_loss_det + (weight_loss_overlap * loss_overlap) + (weight_loss_tv * loss_tv)
    #
    #         # Update generator weights
    #     loss.backward()
    #     opt.step()
    #
    #     # draw bbox at output
    #     if enable_with_bbox and len(bboxes) > 0:
    #         trans_2pilimage = transforms.ToPILImage()
    #         batch = p_img_batch.size()[0]
    #         for b in range(batch):
    #             img_pil = trans_2pilimage(p_img_batch[b].cpu())
    #             img_width = img_pil.size[0]
    #             img_height = img_pil.size[1]
    #             namesfile = 'pytorchYOLOv4/data/coco.names'
    #             class_names = load_class_names(namesfile)
    #             # sample first image
    #             bbox = bboxes[b]
    #             # print("bbox : "+str(bbox))
    #             for box in bbox:
    #                 # print("box size : "+str(box.size()))
    #                 cls_id = box[6].int()
    #                 cls_name = class_names[cls_id]
    #                 cls_conf = box[5]
    #                 if (cls_id == cls_id_attacked):
    #                     if (model_name == "yolov2"):
    #                         x_center = box[0]
    #                         y_center = box[1]
    #                         width = box[2]
    #                         height = box[3]
    #                         left = (x_center.item() - width.item() / 2) * img_width
    #                         right = (x_center.item() + width.item() / 2) * img_width
    #                         top = (y_center.item() - height.item() / 2) * img_height
    #                         bottom = (y_center.item() + height.item() / 2) * img_height
    #                     if (model_name == "yolov5") or (model_name == "yolov3" or (model_name == "fasterrcnn")):
    #                         left = int(box[0] * img_width)
    #                         right = int(box[2] * img_width)
    #                         top = int(box[1] * img_height)
    #                         bottom = int(box[3] * img_height)
    #                     # img with prediction
    #                     draw = ImageDraw.Draw(img_pil)
    #                     shape = [left, top, right, bottom]
    #                     draw.rectangle(shape, outline="red")
    #                     # text
    #                     color = [255, 0, 0]
    #                     font = ImageFont.truetype("cmb10.ttf", int(min(img_width, img_height) / 18))
    #                     sentence = str(cls_name) + " (" + str(round(float(cls_conf), 2)) + ")"
    #                     position = [left, top]
    #                     draw.text(tuple(position), sentence, tuple(color), font=font)
    #             if enable_show_plt:
    #                 # show, debug
    #                 plt.imshow(img_pil)
    #                 plt.show()
    #             trans_2tensor = transforms.ToTensor()
    #             p_img_batch[b] = trans_2tensor(img_pil)
    #     # print("loss_det:",loss)
    #     # st()
    #     return loss_det, loss_overlap, loss_tv, p_img_batch, fake_images, D_loss
    # elif method_num == 3:  # stylegan2
    #     opt.zero_grad()
    #     # latent_shift = torch.clamp(latent_shift,-4,4)
    #     # Generate fake images
    #     rows = 1
    #     # set desired class for conditional GAN
    #     zs = torch.rand(
    #         [rows, generator.latent_size] if type(generator.latent_size) == int else [rows] + generator.latent_size,
    #         device=device)
    #     # Do img = G(z + shift)
    #     # Directly, change z.  z = z + latent_shift
    #     z = latent_shift
    #     interpolation_deformed = generator(z.unsqueeze(0))
    #     # st()
    #     interpolation_deformed = interpolation_deformed.to(device)  # torch.Size([1, 3, 128, 128])
    #     fake_images = (interpolation_deformed + 1) * 0.5
    #     fake_images = torch.clamp(fake_images, 0, 1)
    #     # st()
    #     # images = utils.tensor_to_PIL(
    #     #     generated, pixel_min=args.pixel_min, pixel_max=args.pixel_max)
    #
    #     min_loss_det = 999
    #     for i_fimage in range(1):
    #         if not (enable_clear_output):
    #             adv_batch_t, adv_patch_set, msk_batch = patch_transformer(adv_patch=fake_images[i_fimage],
    #                                                                       lab_batch=label,
    #                                                                       img_size=416,
    #                                                                       patch_mask=[],
    #                                                                       by_rectangle=by_rectangle,
    #                                                                       do_rotate=enable_rotation,
    #                                                                       rand_loc=enable_randomLocation,
    #                                                                       with_black_trans=False,
    #                                                                       scale_rate=patch_scale,
    #                                                                       with_crease=enable_crease,
    #                                                                       with_projection=enable_projection,
    #                                                                       with_rectOccluding=enable_rectOccluding,
    #                                                                       enable_blurred=enable_blurred)
    #             p_img_batch = patch_applier(input_imgs, adv_batch_t)
    #             # label       size : torch.Size([16, 14, 5])
    #             # input_imgs  size : torch.Size([16, 3, 416, 416])
    #             # adv_batch_t size : torch.Size([16, 14, 3, 416, 416])
    #             # p_img_batch size : torch.Size([16, 3, 416, 416])
    #             # adv_patch_set size :torch.Size([3, 300, 300])
    #         else:
    #             p_img_batch = input_imgs
    #         if discriminator is not None:
    #             D_loss = discriminator(fake_images, torch.Tensor([259]).long().cuda())
    #         else:
    #             D_loss = 0
    #         # st()
    #         # loss.
    #         if (model_name == "yolov5"):
    #             max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch,
    #                                                                       cls_id_attacked=cls_id_attacked,
    #                                                                       clear_imgs=input_imgs,
    #                                                                       with_bbox=enable_with_bbox)
    #             # loss_det
    #             # default is multi_score. Here, "max_prob_cls" is "cls*obj"
    #             loss_det = torch.mean(max_prob_obj_cls)
    #             # loss_overlap
    #             loss_overlap = -torch.mean(overlap_score)
    #         if (model_name == "yolov3"):
    #             max_prob_obj, max_prob_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch,
    #                                                                                 cls_id_attacked=cls_id_attacked,
    #                                                                                 clear_imgs=input_imgs,
    #                                                                                 with_bbox=enable_with_bbox)
    #             # loss_det
    #             if multi_score:
    #                 multi = max_prob_obj * max_prob_cls
    #                 loss_det = torch.mean(multi)
    #             else:
    #                 loss_det = torch.mean(max_prob_obj)
    #             # loss_overlap
    #             loss_overlap = -torch.mean(overlap_score)
    #         if (model_name == "yolov2"):
    #             max_prob_obj, max_prob_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch,
    #                                                                                 cls_id_attacked=cls_id_attacked,
    #                                                                                 clear_imgs=input_imgs,
    #                                                                                 with_bbox=enable_with_bbox)
    #             # loss_det
    #             if multi_score:
    #                 multi = max_prob_obj * max_prob_cls
    #                 loss_det = torch.mean(multi)
    #             else:
    #                 loss_det = torch.mean(max_prob_obj)
    #             # loss_overlap
    #             loss_overlap = -torch.mean(overlap_score)
    #         if (model_name == "fasterrcnn"):
    #             max_prob, max_prob, bboxes = fasterrcnn(tensor_image_inputs=p_img_batch, device=device,
    #                                                     cls_id_attacked=cls_id_attacked, threshold=0.5)
    #             # loss_det
    #             loss_det = torch.mean(max_prob)
    #             # no loss_overlap
    #             loss_overlap = torch.tensor(0).to(device)
    #
    #         if (min_loss_det > loss_det):
    #             min_loss_det = loss_det
    #
    #     # loss: total variation
    #     loss_tv = total_variation(fake_images[0])
    #     # st()
    #     # loss
    #     if discriminator is not None:
    #         loss = min_loss_det + (weight_loss_overlap * loss_overlap) + (weight_loss_tv * loss_tv) + 1e-3 * D_loss
    #     else:
    #         loss = min_loss_det + (weight_loss_overlap * loss_overlap) + (weight_loss_tv * loss_tv)
    #
    #         # Update generator weights
    #     loss.backward()
    #     opt.step()
    #
    #     # draw bbox at output
    #     if enable_with_bbox and len(bboxes) > 0:
    #         trans_2pilimage = transforms.ToPILImage()
    #         batch = p_img_batch.size()[0]
    #         for b in range(batch):
    #             img_pil = trans_2pilimage(p_img_batch[b].cpu())
    #             img_width = img_pil.size[0]
    #             img_height = img_pil.size[1]
    #             namesfile = 'pytorchYOLOv4/data/coco.names'
    #             class_names = load_class_names(namesfile)
    #             # sample first image
    #             bbox = bboxes[b]
    #             # print("bbox : "+str(bbox))
    #             for box in bbox:
    #                 # print("box size : "+str(box.size()))
    #                 cls_id = box[6].int()
    #                 cls_name = class_names[cls_id]
    #                 cls_conf = box[5]
    #                 if (cls_id == cls_id_attacked):
    #                     if (model_name == "yolov2"):
    #                         x_center = box[0]
    #                         y_center = box[1]
    #                         width = box[2]
    #                         height = box[3]
    #                         left = (x_center.item() - width.item() / 2) * img_width
    #                         right = (x_center.item() + width.item() / 2) * img_width
    #                         top = (y_center.item() - height.item() / 2) * img_height
    #                         bottom = (y_center.item() + height.item() / 2) * img_height
    #                     if (model_name == "yolov5") or (model_name == "yolov3" or (model_name == "fasterrcnn")):
    #                         left = int(box[0] * img_width)
    #                         right = int(box[2] * img_width)
    #                         top = int(box[1] * img_height)
    #                         bottom = int(box[3] * img_height)
    #                     # img with prediction
    #                     draw = ImageDraw.Draw(img_pil)
    #                     shape = [left, top, right, bottom]
    #                     draw.rectangle(shape, outline="red")
    #                     # text
    #                     color = [255, 0, 0]
    #                     font = ImageFont.truetype("cmb10.ttf", int(min(img_width, img_height) / 18))
    #                     sentence = str(cls_name) + " (" + str(round(float(cls_conf), 2)) + ")"
    #                     position = [left, top]
    #                     draw.text(tuple(position), sentence, tuple(color), font=font)
    #             if enable_show_plt:
    #                 # show, debug
    #                 plt.imshow(img_pil)
    #                 plt.show()
    #             trans_2tensor = transforms.ToTensor()
    #             p_img_batch[b] = trans_2tensor(img_pil)
    #     # print("loss:",loss_det)
    #     return loss_det, loss_overlap, loss_tv, p_img_batch, fake_images, D_loss

# # test
# import numpy as np
# def show(img):
#     npimg = img.numpy()
#     fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), 
#                                     interpolation='nearest')
#     fig.axes.get_xaxis().set_visible(False)
#     fig.axes.get_yaxis().set_visible(False)
# plt.ion()
# plt.figure()
# show(make_grid(p_img_batch.detach().cpu()[:16,:,:,:]))
# plt.ioff()
# plt.show()
# sys.exit()
