import fnmatch
import math
import os
import sys
import time
from operator import itemgetter

import gc
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from imageaug.transforms import Colorspace

import matplotlib.pyplot as plt

from math import pi
import torchgeometry as tgm
from ipdb import set_trace as st

if not (__name__ == 'load_data') and not (__name__ == '__main__'):
    from adversarialYolo.darknet import Darknet
    from adversarialYolo.median_pool import MedianPool2d
    from adversarialYolo.utils import get_rad, get_deg, deg_to_rad, rad_to_deg
else:
    from darknet import Darknet
    from median_pool import MedianPool2d
    from utils import get_rad, get_deg, deg_to_rad, rad_to_deg


# from ..ensemble import enable_rotate

class MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, config):
        super(MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config

    def forward(self, YOLOoutput):
        # get values neccesary for transformation
        if YOLOoutput.dim() == 3:
            YOLOoutput = YOLOoutput.unsqueeze(0)
        batch = YOLOoutput.size(0)
        assert (YOLOoutput.size(1) == (5 + self.num_cls) * 5)
        h = YOLOoutput.size(2)
        w = YOLOoutput.size(3)
        # transform the output tensor from [batch, 425, 19, 19] to [batch, 80, 1805]
        output = YOLOoutput.view(batch, 5, 5 + self.num_cls, h * w)  # [batch, 5, 85, 361]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
        output = output.view(batch, 5 + self.num_cls, 5 * h * w)  # [batch, 85, 1805]
        output_wh = torch.sigmoid(output[:, 2:4, :])  # [batch, 2, 1805]
        output_area = output_wh[:, 0, :] * output_wh[:, 1, :]
        output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 1805]
        output = output[:, 5:5 + self.num_cls, :]  # [batch, 80, 1805]
        # perform softmax to normalize probabilities for object classes to [0,1]
        normal_confs = torch.nn.Softmax(dim=1)(output)  # [batch, 80, 1805] torch.Size([8, 80, 845]). 19,19 -> 13,13
        # we only care for probabilities of the class of interest (person)
        confs_for_class = normal_confs[:, self.cls_id, :]  # [batch, 1805] torch.Size([8, 845]). 19,19 -> 13,13
        # confs_if_object = output_objectness #confs_for_class * output_objectness
        # confs_if_object = confs_for_class * output_objectness
        confs_if_object = self.config.loss_target(output_objectness, confs_for_class)
        # # find the max probability for person
        max_conf_target, max_conf_idx_target = torch.max(confs_if_object,
                                                         dim=1)  # batch, batch. torch.Size([8]), torch.Size([8])
        # max_conf_class, max_conf_idx_class = torch.max(confs_for_class, dim=1)  # batch, batch. torch.Size([8]), torch.Size([8])

        # thre = 0.4
        # zeros = torch.zeros(output_objectness.size()).float().cuda()
        # ones = torch.ones(output_objectness.size()).float().cuda()
        # output_objectness_availalbe = torch.where(output_objectness > thre, output_objectness, zeros)
        # confs_for_class_availalbe   = torch.where(confs_for_class > thre, confs_for_class, zeros)
        # output_area_availalbe       = torch.where((confs_for_class*output_objectness) > thre, output_area, ones)
        # output_score = torch.sum(((output_objectness_availalbe * confs_for_class_availalbe) / output_area_availalbe), dim=1)

        return max_conf_target
        # return output_score


class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, patch_side, printability_file_1, printability_file_2=None):
        super(NPSCalculator, self).__init__()
        self.printability_array_1 = nn.Parameter(self.get_printability_array(printability_file_1, patch_side),
                                                 requires_grad=False)
        if not (printability_file_2 == None):
            self.printability_array_2 = nn.Parameter(self.get_printability_array(printability_file_2, patch_side),
                                                     requires_grad=False)

    def forward(self, adv_patch, key=1):
        # calculate euclidian distance between colors in patch and colors in printability_array 
        # square root of sum of squared difference
        # print("adv_patch size : "+str(adv_patch.size()))  ## torch.Size([3, 300, 300])
        # print("self.printability_array size : "+str(self.printability_array.size()))
        if (key == 1):
            color_dist = (adv_patch - self.printability_array_1 + 0.000001)  ##  torch.Size([30, 3, 300, 300])
        elif (key == 2):
            color_dist = (adv_patch - self.printability_array_2 + 0.000001)  ##  torch.Size([30, 3, 300, 300])
        color_dist = color_dist ** 2  ##                                 torch.Size([30, 3, 300, 300])
        color_dist = torch.sum(color_dist, 1) + 0.000001  ##               torch.Size([30, 300, 300])
        color_dist = torch.sqrt(color_dist)  ##                          torch.Size([30, 300, 300])  
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[
            0]  # test: change prod for min (find distance to closest color)  ##  torch.Size([300, 300])
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod,
                              0)  ##                                                                   torch.Size([300])
        nps_score = torch.sum(nps_score,
                              0)  ##                                                                         torch.Size([])
        return nps_score / torch.numel(adv_patch)

    def get_printability_array(self, printability_file, side):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa


class CSSCalculator(nn.Module):
    """NMSCalculator: calculates the color specified score of a patch.

    Module providing the functionality necessary to calculate the color specified score (CSS) of an adversarial patch.

    """

    def __init__(self, sample_img):
        super(CSSCalculator, self).__init__()
        # self.color_array = nn.Parameter(self.get_color_array(colorSpecified_file, patch_side, patch_unit),requires_grad=False)
        self.color_array = nn.Parameter(self.get_color_array(sample_img), requires_grad=False)

    def forward(self, adv_patch, sample_img):
        n_items_adv_patch = len(adv_patch.size())
        n_items_sample_img = len(sample_img.size())
        if ((n_items_adv_patch == 3) and (n_items_sample_img == 3)):
            # print("sample_img "+str(type(sample_img))+" "+str(sample_img.size())+" "+str(sample_img.dtype))
            self.color_array.data = sample_img.unsqueeze(0)

            # calculate euclidian distance between colors in patch and colors in color_array 
            # square root of sum of squared difference

            # print("adv_patch size: "+str(adv_patch.size()))  ##                  torch.Size([3, 150, 150])
            # print("color_array size: "+str(self.color_array.size()))  ##           torch.Size([1, 3, 300, 300])

            h_target_size = int(adv_patch.size()[-2] / 2)
            w_target_size = int(adv_patch.size()[-1] / 2)
            adv_patch = F.interpolate(adv_patch.unsqueeze(0), (h_target_size, w_target_size))[
                0]  # torch.Size([3, 75, 75])
            self.color_array.data = F.interpolate(self.color_array,
                                                  (h_target_size, w_target_size))  # torch.Size([1, 3, 75, 75])

            color_dist = (adv_patch - self.color_array + 0.000001)  ##               torch.Size([1, 3, 75, 75])
            color_dist = color_dist ** 2  ##                                       torch.Size([1, 3, 75, 75])
            color_dist = torch.sum(color_dist, 1) + 0.000001  ##                     torch.Size([1, 75, 75])
            color_dist = torch.sqrt(color_dist)  ##                                torch.Size([1, 75, 75])
            # only work with the min distance
            color_dist_prod = torch.min(color_dist, 0)[
                0]  # test: change prod for min (find distance to closest color)  ##  torch.Size([75, 75])
            # calculate the nps by summing over all pixels
            nps_score = torch.sum(color_dist_prod,
                                  0)  ##                                                                   torch.Size([75])
            nps_score = torch.sum(nps_score,
                                  0)  ##                                                                         torch.Size([])
            return nps_score / torch.numel(adv_patch)
        elif ((n_items_adv_patch == 5) and (n_items_sample_img == 5)):
            b, f, d, h, w = adv_patch.size()
            adv_patch = adv_patch.view(-1, d, h, w)
            sample_img = sample_img.view(-1, d, h, w)
            self.color_array.data = sample_img.unsqueeze(0)
            color_dist = (adv_patch - self.color_array + 0.000001)  ##               torch.Size([1, 3, 75, 75])
            color_dist = color_dist ** 2  ##                                       torch.Size([1, 3, 75, 75])
            color_dist = torch.sum(color_dist, 2) + 0.000001  ##                     torch.Size([1, 75, 75])
            color_dist = torch.sqrt(color_dist)  ##                                torch.Size([1, 75, 75])
            #
            color_dist_prod = torch.min(color_dist, 0)[
                0]  # test: change prod for min (find distance to closest color)  ##  torch.Size([75, 75])
            # calculate the nps by summing over all pixels
            nps_score = torch.sum(color_dist_prod,
                                  1)  ##                                                                   torch.Size([75])
            nps_score = torch.sum(nps_score,
                                  1)  ##                                                                         torch.Size([])
            nps_score = torch.sum(nps_score, 0)  ##
            return nps_score / torch.numel(adv_patch)

            pass

    def get_color_array(self, sample_img):
        color_array = []
        # from an image to pixel-art
        # sample_img = transforms.ToPILImage()(sample_img.detach().cpu())
        sample_img = sample_img.numpy()[:, :, :]  # torch to numpy array
        # print("sample_img size: "+str(sample_img.shape))  ##  sample_img size: (3, 300, 300)
        color_array.append(sample_img)
        # #
        # sample_img_ = np.einsum('kli->lik', sample_img)
        # plt.imshow(sample_img_)
        # plt.show()

        # print("color_array size: "+str(np.array(color_array).shape))  ##  color_array size: (1, 3, 300, 300)
        color_array = np.asarray(color_array)
        # print("color_array size: "+str(np.array(color_array).shape))  ##  color_array size: (1, 3, 300, 300)
        color_array = np.float32(color_array)
        # print("color_array size: "+str(np.array(color_array).shape))  ##  color_array size: (1, 3, 300, 300)
        pa = torch.from_numpy(color_array)
        # print("pa size: "+str(pa.size()))  ##                           pa size: torch.Size([1, 3, 300, 300])
        return pa


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)


class PatchSimpleTransformer(nn.Module):
    def __init__(self):
        super(PatchSimpleTransformer, self).__init__()
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)

    def rect_occluding(self, num_rect=1, n_batch=8, n_feature=14, patch_size=300, with_cuda=True):
        if (with_cuda):
            device = 'cuda:0'
        else:
            device = 'cpu'
        tensor_img = torch.full((3, patch_size, patch_size), 0.0).to(device)
        for ttt in range(num_rect):
            xs = torch.randint(0, int(patch_size / 2), (1,))[0]
            xe = torch.randint(xs,
                               torch.min(torch.tensor(tensor_img.size()[-1]), xs + int(patch_size / 2)),
                               (1,))[0]
            ys = torch.randint(0, int(patch_size / 2), (1,))[0]
            ye = torch.randint(ys,
                               torch.min(torch.tensor(tensor_img.size()[-1]), ys + int(patch_size / 2)),
                               (1,))[0]
            tensor_img[:, xs:xe, ys:ye] = 0.5
        tensor_img_batch = tensor_img.unsqueeze(0)  ##  torch.Size([1, 3, 300, 300])
        tensor_img_batch = tensor_img_batch.expand(n_batch, n_feature, -1, -1, -1)  ##  torch.Size([8, 14, 3, 300, 300])
        return tensor_img_batch.to(device)

    def deg_to_rad(self, deg):
        return torch.tensor(deg * pi / 180.0).float().cuda()

    def rad_to_deg(self, rad):
        return torch.tensor(rad * 180.0 / pi).float().cuda()

    def get_warpR(self, anglex, angley, anglez, fov, w, h):
        fov = torch.tensor(fov).float().cuda()
        w = torch.tensor(w).float().cuda()
        h = torch.tensor(h).float().cuda()
        z = torch.sqrt(w ** 2 + h ** 2) / 2 / torch.tan(deg_to_rad(fov / 2)).float().cuda()
        rx = torch.tensor([[1, 0, 0, 0],
                           [0, torch.cos(deg_to_rad(anglex)), -torch.sin(deg_to_rad(anglex)), 0],
                           [0, -torch.sin(deg_to_rad(anglex)), torch.cos(deg_to_rad(anglex)), 0, ],
                           [0, 0, 0, 1]]).float().cuda()
        ry = torch.tensor([[torch.cos(deg_to_rad(angley)), 0, torch.sin(deg_to_rad(angley)), 0],
                           [0, 1, 0, 0],
                           [-torch.sin(deg_to_rad(angley)), 0, torch.cos(deg_to_rad(angley)), 0, ],
                           [0, 0, 0, 1]]).float().cuda()
        rz = torch.tensor([[torch.cos(deg_to_rad(anglez)), torch.sin(deg_to_rad(anglez)), 0, 0],
                           [-torch.sin(deg_to_rad(anglez)), torch.cos(deg_to_rad(anglez)), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]).float().cuda()
        r = torch.matmul(torch.matmul(rx, ry), rz)
        pcenter = torch.tensor([h / 2, w / 2, 0, 0]).float().cuda()
        p1 = torch.tensor([0, 0, 0, 0]).float().cuda() - pcenter
        p2 = torch.tensor([w, 0, 0, 0]).float().cuda() - pcenter
        p3 = torch.tensor([0, h, 0, 0]).float().cuda() - pcenter
        p4 = torch.tensor([w, h, 0, 0]).float().cuda() - pcenter
        dst1 = torch.matmul(r, p1)
        dst2 = torch.matmul(r, p2)
        dst3 = torch.matmul(r, p3)
        dst4 = torch.matmul(r, p4)
        list_dst = [dst1, dst2, dst3, dst4]
        org = torch.tensor([[0, 0],
                            [w, 0],
                            [0, h],
                            [w, h]]).float().cuda()
        dst = torch.zeros((4, 2)).float().cuda()
        for i in range(4):
            dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
            dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]
        org = org.unsqueeze(0)
        dst = dst.unsqueeze(0)
        warpR = tgm.get_perspective_transform(org, dst).float().cuda()
        return warpR

    def warping(self, input_tensor_img, wrinkle_p=15):
        C, H, W = input_tensor_img.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, H, W)
        yy = yy.view(1, H, W)
        grid = torch.cat((xx, yy), 0).float()  # torch.Size([2, H, W])
        # print("grid "+str(grid.shape)+" : \n"+str(grid))
        grid = grid.view(2, -1)  # torch.Size([2, H*W])
        grid = grid.permute(1, 0)  # torch.Size([H*W, 2])
        perturbed_mesh = grid

        # nv = np.random.randint(20) - 1
        nv = wrinkle_p
        for k in range(nv):
            # Choosing one vertex randomly
            vidx = np.random.randint(grid.shape[0])
            vtex = grid[vidx, :]
            # Vector between all vertices and the selected one
            xv = perturbed_mesh - vtex
            # Random movement 
            mv = (np.random.rand(1, 2) - 0.5) * 20
            hxv = np.zeros((np.shape(xv)[0], np.shape(xv)[1] + 1))
            hxv[:, :-1] = xv
            hmv = np.tile(np.append(mv, 0), (np.shape(xv)[0], 1))
            d = np.cross(hxv, hmv)
            d = np.absolute(d[:, 2])
            # print("d "+str(d.shape)+" :\n"+str(d))
            d = d / (np.linalg.norm(mv, ord=2))
            wt = d
            curve_type = np.random.rand(1)
            if curve_type > 0.3:
                alpha = np.random.rand(1) * 50 + 50
                wt = alpha / (wt + alpha)
            else:
                alpha = np.random.rand(1) + 1
                wt = 1 - (wt / 100) ** alpha
            msmv = mv * np.expand_dims(wt, axis=1)
            perturbed_mesh = perturbed_mesh + msmv

        perturbed_mesh_2 = perturbed_mesh.permute(1, 0)
        max_x = torch.max(perturbed_mesh_2[0])
        min_x = torch.min(perturbed_mesh_2[0])
        # print("max_x : "+str(max_x)+" / min_x : "+str(min_x))
        max_y = torch.max(perturbed_mesh_2[1])
        min_y = torch.min(perturbed_mesh_2[1])
        # print("max_y : "+str(max_y)+" / min_y : "+str(min_y))
        perturbed_mesh_2[0, :] = (W - 1) * (perturbed_mesh_2[0, :] - min_x) / (max_x - min_x)
        perturbed_mesh_2[1, :] = (H - 1) * (perturbed_mesh_2[1, :] - min_y) / (max_y - min_y)
        # max_x = torch.max(perturbed_mesh_2[0])
        # min_x = torch.min(perturbed_mesh_2[0])
        # print("max_x : "+str(max_x)+" / min_x : "+str(min_x))
        # max_y = torch.max(perturbed_mesh_2[1])
        # min_y = torch.min(perturbed_mesh_2[1])
        # print("max_y : "+str(max_y)+" / min_y : "+str(min_y))
        perturbed_mesh_2 = perturbed_mesh_2.contiguous().view(-1, H, W).float()
        # print("perturbed_mesh_2 dtype : "+str(perturbed_mesh_2.data.type()))
        # print("perturbed_mesh_2 "+str(perturbed_mesh_2.shape)+" : \n"+str(perturbed_mesh_2))

        vgrid = perturbed_mesh_2.unsqueeze(0).cuda()
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / (W - 1) - 1.0  # max(W-1,1)
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / (H - 1) - 1.0  # max(H-1,1)
        vgrid = vgrid.permute(0, 2, 3, 1).cuda()
        input_tensor_img_b = input_tensor_img.unsqueeze(0).cuda()
        output = F.grid_sample(input_tensor_img_b, vgrid, align_corners=True)  # torch.Size([1, 3, H, W])
        return output[0]

    def forward(self, adv_patch, lab_batch, img_size, patch_mask=[], do_rotate=True, rand_loc=True,
                with_black_trans=False, scale_rate=0.2, with_crease=False, with_projection=False,
                with_rectOccluding=False):
        if (with_crease):
            # warping
            adv_patch = self.warping(adv_patch)  # torch.Size([3, H, W])
            # print("adv_patch "+str(adv_patch.size())+"  "+str(adv_patch.dtype))

        #
        # adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        # print("adv_patch medianpooler size: "+str(adv_patch.size())) ## torch.Size([1, 3, 300, 300])
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)  ##  torch.Size([1, 1, 3, 300, 300])
        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1,
                                     -1)  ##  torch.Size([8, 14, 3, 300, 300])
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

        if not (len(patch_mask) == 0):
            ## mask size : torch.Size([3, 300, 300])
            patch_mask = patch_mask.unsqueeze(0)  ## mask size : torch.Size([1, 3, 300, 300])
            mask_batch = patch_mask.expand(lab_batch.size(0), lab_batch.size(1), -1, -1,
                                           -1)  ## mask size : torch.Size([8, 14, 3, 300, 300])

        # Contrast, brightness and noise transforms

        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()
        # print("contrast size : "+str(contrast.size()))  ##  contrast size : torch.Size([8, 14, 3, 300, 300])

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()
        # print("brightness size : "+str(brightness.size())) ##  brightness size : torch.Size([8, 14, 3, 300, 300])

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
        # print("noise size : "+str(noise.size()))  ##  noise size : torch.Size([8, 14, 3, 300, 300])

        # # Apply contrast/brightness/noise, clamp

        # print("adv_patch  : "+str(adv_patch.is_cuda))
        # print("adv_batch  : "+str(adv_batch.is_cuda))
        # print("contrast   : "+str(contrast.is_cuda))
        # print("brightness : "+str(brightness.is_cuda))
        # print("noise      : "+str(noise.is_cuda))

        adv_batch = adv_batch * contrast + brightness + noise
        if not (len(patch_mask) == 0):
            adv_batch = adv_batch * mask_batch
        if (with_rectOccluding):
            rect_occluder = self.rect_occluding(num_rect=2, n_batch=adv_batch.size()[0], n_feature=adv_batch.size()[1],
                                                patch_size=adv_batch.size()[-1])
            adv_batch = torch.where((rect_occluder == 0), adv_batch, rect_occluder)

        if (with_black_trans):
            adv_batch = torch.clamp(adv_batch, 0.0, 0.99999)
        else:
            adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)
        adv_patch_set = adv_batch[0, 0]

        def resize_rotate(adv_batch):

            if (with_black_trans):
                adv_batch = torch.clamp(adv_batch, 0.0, 0.99999)
            else:
                adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

            # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
            cls_ids = torch.narrow(lab_batch, 2, 0, 1)  # torch.Size([8, 14, 1])
            cls_mask = cls_ids.expand(-1, -1, 3)  # torch.Size([8, 14, 3])
            cls_mask = cls_mask.unsqueeze(-1)  # torch.Size([8, 14, 3, 1])
            cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))  # torch.Size([8, 14, 3, 300])
            cls_mask = cls_mask.unsqueeze(-1)  # torch.Size([8, 14, 3, 300, 1])
            cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))  # torch.Size([8, 14, 3, 300, 300])
            msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask  # torch.Size([8, 14, 3, 300, 300])

            # Pad patch and mask to image dimensions
            # Determine size of padding
            pad = (img_size - msk_batch.size(-1)) / 2  # (416-300) / 2 = 58
            # print("pad : "+str(pad))
            mypad = nn.ConstantPad2d((int(pad), int(pad), int(pad), int(pad)), 0)
            # print("adv_batch size : "+str(adv_batch.size()))
            adv_batch = mypad(adv_batch)  # adv_batch size : torch.Size([8, 14, 3, 416, 416])
            msk_batch = mypad(msk_batch)  # adv_batch size : torch.Size([8, 14, 3, 416, 416])
            # print("adv_batch size : "+str(adv_batch.size()))

            # Rotation and rescaling transforms
            anglesize = (lab_batch.size(0) * lab_batch.size(1))  # 8*14 = 112
            if do_rotate:
                angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)  # torch.Size([112])
            else:
                angle = torch.cuda.FloatTensor(anglesize).fill_(0)

            # Resizes and rotates
            current_patch_size = adv_patch.size(-1)
            lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)  # torch.Size([8, 14, 5])
            lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
            lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
            lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
            lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
            target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(scale_rate)) ** 2) + (
                        (lab_batch_scaled[:, :, 4].mul(scale_rate)) ** 2))  # torch.Size([8, 14])
            target_x = lab_batch[:, :, 1].view(np.prod(batch_size))  # torch.Size([112]) 8*14
            target_y = lab_batch[:, :, 2].view(np.prod(batch_size))  # torch.Size([112]) 8*14
            targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))  # torch.Size([112]) 8*14
            targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))  # torch.Size([112]) 8*14
            if (rand_loc):
                off_x = targetoff_x * (torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))
                target_x = target_x + off_x
                off_y = targetoff_y * (torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))
                target_y = target_y + off_y
            target_y = target_y - 0.05
            # print("current_patch_size : "+str(current_patch_size))
            # print("target_size        : "+str(target_size.size()))
            # print("target_size        : "+str(target_size))
            scale = target_size / current_patch_size  # torch.Size([8, 14])
            scale = scale.view(anglesize)  # torch.Size([112]) 8*14
            # print("scale : "+str(scale))

            s = adv_batch.size()
            adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])  # torch.Size([112, 3, 416, 416])
            msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])  # torch.Size([112, 3, 416, 416])

            tx = (-target_x + 0.5) * 2
            ty = (-target_y + 0.5) * 2
            sin = torch.sin(angle)
            cos = torch.cos(angle)

            # Theta = rotation,rescale matrix
            theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)  # torch.Size([112, 2, 3])
            theta[:, 0, 0] = cos / scale
            theta[:, 0, 1] = sin / scale
            theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
            theta[:, 1, 0] = -sin / scale
            theta[:, 1, 1] = cos / scale
            theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

            # print(tx)
            # print(theta[:, 0, 2])
            # print(1*cos/scale)
            # print(-1*cos/scale)

            b_sh = adv_batch.shape  # b_sh = torch.Size([112, 3, 416, 416])
            grid = F.affine_grid(theta, adv_batch.shape)  # torch.Size([112, 416, 416, 2])

            adv_batch_t = F.grid_sample(adv_batch, grid)  # torch.Size([112, 3, 416, 416])
            msk_batch_t = F.grid_sample(msk_batch, grid)  # torch.Size([112, 3, 416, 416])

            # print("grid : "+str(grid[0,200:300,200:300,:]))

            # msk_batch_t_r = msk_batch_t[:,0,:,:]
            # msk_batch_t_g = msk_batch_t[:,0,:,:]
            # msk_batch_t_b = msk_batch_t[:,0,:,:]
            # for t in range(msk_batch_t.size()[0]):
            #     dx = int(grid[t,0,0,0])
            #     dx2 = int(grid[t,400,400,0])
            #     dy = int(grid[t,0,0,1])
            #     dy2 = int(grid[t,400,400,1])
            #     msk_batch_t[t,0,dx:dx2,dy:dy2] = 0
            #     msk_batch_t[t,1,dx:dx2,dy:dy2] = 0
            #     msk_batch_t[t,2,dx:dx2,dy:dy2] = 0

            # # angle 2
            # tx = (-target_x+0.5)*2
            # ty = (-target_y+0.5)*2
            # sin = torch.sin(angle)
            # cos = torch.cos(angle)        

            # # Theta = rotation,rescale matrix
            # theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)  # torch.Size([112, 2, 3])
            # theta[:, 0, 0] = cos/scale
            # theta[:, 0, 1] = sin/scale
            # theta[:, 0, 2] = 0
            # theta[:, 1, 0] = -sin/scale
            # theta[:, 1, 1] = cos/scale
            # theta[:, 1, 2] = 0

            '''
            # Theta2 = translation matrix
            theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
            theta2[:, 0, 0] = 1
            theta2[:, 0, 1] = 0
            theta2[:, 0, 2] = (-target_x + 0.5) * 2
            theta2[:, 1, 0] = 0
            theta2[:, 1, 1] = 1
            theta2[:, 1, 2] = (-target_y + 0.5) * 2

            grid2 = F.affine_grid(theta2, adv_batch.shape)
            adv_batch_t = F.grid_sample(adv_batch_t, grid2)
            msk_batch_t = F.grid_sample(msk_batch_t, grid2)

            '''
            adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])  # torch.Size([8, 14, 3, 416, 416])
            msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])  # torch.Size([8, 14, 3, 416, 416])

            if (with_black_trans):
                adv_batch_t = torch.clamp(adv_batch_t, 0.0, 0.99999)
            else:
                adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.99999)
            # img = msk_batch_t[0, 0, :, :, :].detach().cpu()
            # img = transforms.ToPILImage()(img)
            # img.show()
            # exit()

            # output: torch.Size([8, 14, 3, 416, 416]), torch.Size([8, 14, 3, 416, 416])
            # return adv_batch_t * msk_batch_t, (adv_batch_t * msk_batch_t0), (adv_batch_t * msk_batch_t1), (adv_batch_t * msk_batch_t2),  (adv_batch_t * msk_batch_t3), adv_batch_t, msk_batch_t
            return (adv_batch_t * msk_batch_t), msk_batch_t

        # adv_batch_masked, adv_batch_masked0, adv_batch_masked1, adv_batch_masked3, adv_batch_masked4, adv_batch_t, msk_batch_t = resize_rotate(adv_batch)
        adv_batch_masked, msk_batch = resize_rotate(
            adv_batch)  # adv_batch torch.Size([8, 7, 3, 150, 150])   adv_batch_masked torch.Size([8, 7, 3, 416, 416])

        if (with_projection):
            adv_batch = adv_batch_masked
            # # Rotating a Image
            b, f, c, h, w = adv_batch.size()
            adv_batch = adv_batch.view(b * f, c, h, w)
            # print("adv_batch "+str(adv_batch.size())+"  "+str(adv_batch.dtype))
            batch, channel, width, height = adv_batch.size()
            padding_borader = torch.nn.ZeroPad2d(50)
            input_ = padding_borader(adv_batch)
            # print("input_ "+str(input_.size())+"  "+str(input_.dtype))
            angle = np.random.randint(low=-50, high=51)
            mat = self.get_warpR(anglex=0, angley=angle, anglez=0, fov=42, w=width, h=height)
            mat = mat.expand(batch, -1, -1, -1)
            # print("image  "+str(self.image.dtype)+"  "+str(self.image.size()))
            # print("input_ "+str(input_.dtype)+"  "+str(input_.size()))
            # print("mat    "+str(mat.dtype)+"  "+str(mat.size()))
            adv_batch = tgm.warp_perspective(input_, mat, (input_.size()[-2], input_.size()[-1]))
            # print("adv_batch "+str(adv_batch.size())+"  "+str(adv_batch.dtype))
            adv_batch = adv_batch.view(b, f, c, input_.size()[-2], input_.size()[-1])
            # print("adv_batch "+str(adv_batch.size())+"  "+str(adv_batch.dtype))
            ##
            # Pad patch and mask to image dimensions
            # Determine size of padding
            pad = (img_size - adv_batch.size(-1)) / 2  # (416-300) / 2 = 58
            mypad = nn.ConstantPad2d((int(pad), int(pad), int(pad), int(pad)), 0)
            adv_batch = mypad(adv_batch)  # adv_batch size : torch.Size([8, 14, 3, 416, 416])
            adv_batch_masked = adv_batch

        # adv_batch_masked = torch.clamp(adv_batch_masked, 0.0, 0.99999)

        # return adv_batch_masked, adv_batch_masked0, adv_batch_masked1, adv_batch_masked3, adv_batch_masked4, adv_batch_t, msk_batch_t, adv_patch_set
        return adv_batch_masked, adv_patch_set, msk_batch


class PatchTransformer_out_of_bbox(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self, bias_coordinate):
        super(PatchTransformer_out_of_bbox, self).__init__()
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)
        self.bias_coordinate = bias_coordinate
        '''
        kernel = torch.cuda.FloatTensor([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,3,-1,-1)
        '''

    def rect_occluding(self, num_rect=1, n_batch=8, n_feature=14, patch_size=300, with_cuda=True):
        if (with_cuda):
            device = 'cuda:0'
        else:
            device = 'cpu'
        tensor_img = torch.full((3, patch_size, patch_size), 0.0).to(device)
        for ttt in range(num_rect):
            xs = torch.randint(0, int(patch_size / 2), (1,))[0]
            xe = torch.randint(xs,
                               torch.min(torch.tensor(tensor_img.size()[-1]), xs + int(patch_size / 2)),
                               (1,))[0]
            ys = torch.randint(0, int(patch_size / 2), (1,))[0]
            ye = torch.randint(ys,
                               torch.min(torch.tensor(tensor_img.size()[-1]), ys + int(patch_size / 2)),
                               (1,))[0]
            tensor_img[:, xs:xe, ys:ye] = 0.5
        tensor_img_batch = tensor_img.unsqueeze(0)  ##  torch.Size([1, 3, 300, 300])
        tensor_img_batch = tensor_img_batch.expand(n_batch, n_feature, -1, -1, -1)  ##  torch.Size([8, 14, 3, 300, 300])
        return tensor_img_batch.to(device)

    def forward(self, adv_patch, lab_batch, img_size, patch_mask=[], cls_id_attacked=11, by_rectangle=False,
                do_rotate=True, rand_loc=True,
                with_black_trans=False, scale_rate=0.2, with_crease=False, with_projection=False,
                with_rectOccluding=False, enable_empty_patch=False, enable_no_random=False, enable_blurred=True,
                position='bottom'):
        # try:
        #     if cls_id_attacked == 0 and num_of_round > 0:
        #         raise Exception
        # except Exception as e:
        #     print(
        #         "Please choose a appropriate number of patches (like num_of_patches = 1) if you want to attack person.")
        #     sys.exit(1)
        #
        # try:
        #     if cls_id_attacked == 11 and num_of_round > 1:
        #         raise Exception
        # except Exception as e:
        #     print(
        #         "Please choose a appropriate number of patches (like num_of_patches = 2) if you want to attack stop sign.")
        #     sys.exit(1)
        # torch.set_printoptions(edgeitems=sys.maxsize)
        # print("adv_patch size: "+str(adv_patch.size()))
        # patch_size = adv_patch.size(2)

        # init adv_patch. torch.Size([3, 128, 128])
        adv_patch_size = adv_patch.size()[1:3]
        if adv_patch_size[0] > img_size or adv_patch_size[1] > img_size:  # > img_size(416)
            adv_patch = adv_patch.unsqueeze(0)
            if cls_id_attacked == 0:
                adv_patch = F.interpolate(adv_patch, size=img_size)
            if cls_id_attacked == 11:
                adv_patch = F.interpolate(adv_patch, size=(int(img_size / 2), img_size))
            adv_patch = adv_patch[0]

        # st()
        # np.save('gg', adv_batch.cpu().detach().numpy())
        # gg=np.load('gg.npy')   np.argwhere(gg!=adv_batch.cpu().detach().numpy())
        def deg_to_rad(deg):
            return torch.tensor(deg * pi / 180.0).float().cuda()

        def rad_to_deg(rad):
            return torch.tensor(rad * 180.0 / pi).float().cuda()

        def get_warpR(anglex, angley, anglez, fov, w, h):
            fov = torch.tensor(fov).float().cuda()
            w = torch.tensor(w).float().cuda()
            h = torch.tensor(h).float().cuda()
            z = torch.sqrt(w ** 2 + h ** 2) / 2 / torch.tan(deg_to_rad(fov / 2)).float().cuda()
            rx = torch.tensor([[1, 0, 0, 0],
                               [0, torch.cos(deg_to_rad(anglex)), -torch.sin(deg_to_rad(anglex)), 0],
                               [0, -torch.sin(deg_to_rad(anglex)), torch.cos(deg_to_rad(anglex)), 0, ],
                               [0, 0, 0, 1]]).float().cuda()
            ry = torch.tensor([[torch.cos(deg_to_rad(angley)), 0, torch.sin(deg_to_rad(angley)), 0],
                               [0, 1, 0, 0],
                               [-torch.sin(deg_to_rad(angley)), 0, torch.cos(deg_to_rad(angley)), 0, ],
                               [0, 0, 0, 1]]).float().cuda()
            rz = torch.tensor([[torch.cos(deg_to_rad(anglez)), torch.sin(deg_to_rad(anglez)), 0, 0],
                               [-torch.sin(deg_to_rad(anglez)), torch.cos(deg_to_rad(anglez)), 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]]).float().cuda()
            r = torch.matmul(torch.matmul(rx, ry), rz)
            pcenter = torch.tensor([h / 2, w / 2, 0, 0]).float().cuda()
            p1 = torch.tensor([0, 0, 0, 0]).float().cuda() - pcenter
            p2 = torch.tensor([w, 0, 0, 0]).float().cuda() - pcenter
            p3 = torch.tensor([0, h, 0, 0]).float().cuda() - pcenter
            p4 = torch.tensor([w, h, 0, 0]).float().cuda() - pcenter
            dst1 = torch.matmul(r, p1)
            dst2 = torch.matmul(r, p2)
            dst3 = torch.matmul(r, p3)
            dst4 = torch.matmul(r, p4)
            list_dst = [dst1, dst2, dst3, dst4]
            org = torch.tensor([[0, 0],
                                [w, 0],
                                [0, h],
                                [w, h]]).float().cuda()
            dst = torch.zeros((4, 2)).float().cuda()
            for i in range(4):
                dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
                dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]
            org = org.unsqueeze(0)
            dst = dst.unsqueeze(0)
            warpR = tgm.get_perspective_transform(org, dst).float().cuda()
            return warpR

        ## get y gray
        # adv_patch_yuv = Colorspace("rgb", "yuv")(adv_patch).cuda()
        # y = adv_patch_yuv[0].unsqueeze(0)
        # adv_patch_new_y_gray = torch.cat((y,y,y), 0).cuda()
        ## get   gray
        # y = (0.2989 * adv_patch[0] + 0.5870 * adv_patch[1] + 0.1140 * adv_patch[2]).unsqueeze(0)
        # adv_patch_new_y_gray = torch.cat((y,y,y), 0).cuda()
        # adv_patch = adv_patch_new_y_gray

        def warping(input_tensor_img, wrinkle_p=15):
            C, H, W = input_tensor_img.size()
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float()  # torch.Size([2, H, W])
            # print("grid "+str(grid.shape)+" : \n"+str(grid))
            grid = grid.view(2, -1)  # torch.Size([2, H*W])
            grid = grid.permute(1, 0)  # torch.Size([H*W, 2])
            perturbed_mesh = grid

            # nv = np.random.randint(20) - 1
            nv = wrinkle_p
            for k in range(nv):
                # Choosing one vertex randomly
                vidx = np.random.randint(grid.shape[0])
                vtex = grid[vidx, :]
                # Vector between all vertices and the selected one
                xv = perturbed_mesh - vtex
                # Random movement
                mv = (np.random.rand(1, 2) - 0.5) * 20
                hxv = np.zeros((np.shape(xv)[0], np.shape(xv)[1] + 1))
                hxv[:, :-1] = xv
                hmv = np.tile(np.append(mv, 0), (np.shape(xv)[0], 1))
                d = np.cross(hxv, hmv)
                d = np.absolute(d[:, 2])
                # print("d "+str(d.shape)+" :\n"+str(d))
                d = d / (np.linalg.norm(mv, ord=2))
                wt = d
                curve_type = np.random.rand(1)
                if curve_type > 0.3:
                    alpha = np.random.rand(1) * 50 + 50
                    wt = alpha / (wt + alpha)
                else:
                    alpha = np.random.rand(1) + 1
                    wt = 1 - (wt / 100) ** alpha
                msmv = mv * np.expand_dims(wt, axis=1)
                perturbed_mesh = perturbed_mesh + msmv

            perturbed_mesh_2 = perturbed_mesh.permute(1, 0)
            max_x = torch.max(perturbed_mesh_2[0])
            min_x = torch.min(perturbed_mesh_2[0])
            # print("max_x : "+str(max_x)+" / min_x : "+str(min_x))
            max_y = torch.max(perturbed_mesh_2[1])
            min_y = torch.min(perturbed_mesh_2[1])
            # print("max_y : "+str(max_y)+" / min_y : "+str(min_y))
            perturbed_mesh_2[0, :] = (W - 1) * (perturbed_mesh_2[0, :] - min_x) / (max_x - min_x)
            perturbed_mesh_2[1, :] = (H - 1) * (perturbed_mesh_2[1, :] - min_y) / (max_y - min_y)
            # max_x = torch.max(perturbed_mesh_2[0])
            # min_x = torch.min(perturbed_mesh_2[0])
            # print("max_x : "+str(max_x)+" / min_x : "+str(min_x))
            # max_y = torch.max(perturbed_mesh_2[1])
            # min_y = torch.min(perturbed_mesh_2[1])
            # print("max_y : "+str(max_y)+" / min_y : "+str(min_y))
            perturbed_mesh_2 = perturbed_mesh_2.contiguous().view(-1, H, W).float()
            # print("perturbed_mesh_2 dtype : "+str(perturbed_mesh_2.data.type()))
            # print("perturbed_mesh_2 "+str(perturbed_mesh_2.shape)+" : \n"+str(perturbed_mesh_2))

            vgrid = perturbed_mesh_2.unsqueeze(0).cuda()
            vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / (W - 1) - 1.0  # max(W-1,1)
            vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / (H - 1) - 1.0  # max(H-1,1)
            vgrid = vgrid.permute(0, 2, 3, 1).cuda()
            input_tensor_img_b = input_tensor_img.unsqueeze(0).cuda()
            output = F.grid_sample(input_tensor_img_b, vgrid, align_corners=True)  # torch.Size([1, 3, H, W])
            return output[0]

        if (with_crease):
            # warping
            adv_patch = warping(adv_patch)  # torch.Size([3, H, W])
            # print("adv_patch "+str(adv_patch.size())+"  "+str(adv_patch.dtype))

        #
        # adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        if (enable_blurred):
            adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        else:
            adv_patch = adv_patch.unsqueeze(0)
        # print("adv_patch medianpooler size: "+str(adv_patch.size())) ## torch.Size([1, 3, 300, 300])
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)  ##  torch.Size([1, 1, 3, 300, 300])
        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1,
                                     -1)  ##  torch.Size([8, 14, 3, 300, 300])
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

        if not (len(patch_mask) == 0):
            ## mask size : torch.Size([3, 300, 300])
            patch_mask = patch_mask.unsqueeze(0)  ## mask size : torch.Size([1, 3, 300, 300])
            mask_batch = patch_mask.expand(lab_batch.size(0), lab_batch.size(1), -1, -1,
                                           -1)  ## mask size : torch.Size([8, 14, 3, 300, 300])

        # Contrast, brightness and noise transforms

        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()
        # print("contrast size : "+str(contrast.size()))  ##  contrast size : torch.Size([8, 14, 3, 300, 300])

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()
        # print("brightness size : "+str(brightness.size())) ##  brightness size : torch.Size([8, 14, 3, 300, 300])

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
        # print("noise size : "+str(noise.size()))  ##  noise size : torch.Size([8, 14, 3, 300, 300])
        # print(noise[0,0,0,:10,0])
        # # Apply contrast/brightness/noise, clamp

        # print("adv_patch  : "+str(adv_patch.is_cuda))
        # print("adv_batch  : "+str(adv_batch.is_cuda))
        # print("contrast   : "+str(contrast.is_cuda))
        # print("brightness : "+str(brightness.is_cuda))
        # print("noise      : "+str(noise.is_cuda))

        ## adv_patch 已經模糊
        if enable_no_random and not (enable_empty_patch):
            adv_batch = adv_batch
        if not (enable_no_random) and not (enable_empty_patch):
            adv_batch = adv_batch * contrast + brightness + noise
        if not (len(patch_mask) == 0):
            adv_batch = adv_batch * mask_batch
        if (with_rectOccluding):
            rect_occluder = self.rect_occluding(num_rect=2, n_batch=adv_batch.size()[0], n_feature=adv_batch.size()[1],
                                                patch_size=adv_batch.size()[-1])
            adv_batch = torch.where((rect_occluder == 0), adv_batch, rect_occluder)

        # # get   gray
        # # print("adv_batch size: "+str(adv_batch.size()))  ##  torch.Size([8, 14, 3, 300, 300])
        # # adv_batch = adv_batch.cpu()
        # adv_batch_r = adv_batch[:,:,0,:,:]  ##  torch.Size([3, 300, 300])
        # adv_batch_g = adv_batch[:,:,1,:,:]  ##  torch.Size([3, 300, 300])
        # adv_batch_b = adv_batch[:,:,2,:,:]  ##  torch.Size([3, 300, 300])
        # # print("adv_batch_r size: "+str(adv_batch_r.size()))  ##  torch.Size([8, 14, 300, 300])
        # y = (0.2989 * adv_batch_r + 0.5870 * adv_batch_g + 0.1140 * adv_batch_b)
        # y = y.unsqueeze(2)
        # # print("y size: "+str(y.size()))  ##  torch.Size([8, 14, 3, 300, 300])
        # adv_batch_new_y_gray = torch.cat((y,y,y), 2).cuda()
        # adv_batch = adv_batch_new_y_gray
        # # print("adv_batch size: "+str(adv_batch.size()))  ##  torch.Size([8, 14, 3, 300, 300])
        # # adv_batch = adv_batch.cuda()

        #
        if (with_black_trans):
            adv_batch = torch.clamp(adv_batch, 0.0, 0.99999)
        else:
            adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)
        adv_patch_set = adv_batch[0, 0]

        # ## split img
        # # print("adv_batch size : "+str(adv_batch.size()))  ##  torch.Size([8, 14, 3, 300, 300])
        # adv_batch_units = []
        # split_side = 2
        # split_step = patch_size / split_side
        # # print("split_step : "+str(split_step))
        # for stx in range(0, split_side):
        #     for sty in range(0, split_side):
        #         x_s = int(0 + stx*split_step)
        #         y_s = int(0 + sty*split_step)
        #         x_e = int(x_s+split_step)
        #         y_e = int(y_s+split_step)
        #         adv_batch_unit = adv_batch[:,:,:, x_s:x_e, y_s:y_e].cuda()
        #         adv_batch_zeroes = torch.zeros(adv_batch.size()).cuda()
        #         adv_batch_zeroes[:,:,:, x_s:x_e, y_s:y_e] = adv_batch_unit
        #         adv_batch_unit = adv_batch_zeroes
        #         adv_batch_units.append(adv_batch_unit)

        def resize_rotate(adv_batch, by_rectangle=False, cls_id_attacked=11, position='bottom'):
            if cls_id_attacked == 0:
                if (with_black_trans):
                    adv_batch = torch.clamp(adv_batch, 0.0, 0.99999)
                else:
                    adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

                # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
                cls_ids = torch.narrow(lab_batch, 2, 0, 1)  # torch.Size([8, 14, 1])
                cls_mask = cls_ids.expand(-1, -1, 3)  # torch.Size([8, 14, 3])
                cls_mask = cls_mask.unsqueeze(-1)  # torch.Size([8, 14, 3, 1])
                cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))  # torch.Size([8, 14, 3, 300])
                cls_mask = cls_mask.unsqueeze(-1)  # torch.Size([8, 14, 3, 300, 1])
                cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))  # torch.Size([8, 14, 3, 300, 300])
                msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(
                    1) - cls_mask  # torch.Size([8, 14, 3, 300, 300])

                # Pad patch and mask to image dimensions
                # Determine size of padding
                pad = (img_size - msk_batch.size(-1)) / 2  # (416-300) / 2 = 58
                # print("pad : "+str(pad))
                mypad = nn.ConstantPad2d((int(pad), int(pad), int(pad), int(pad)), 0)
                # print("adv_batch size : "+str(adv_batch.size()))
                adv_batch = mypad(adv_batch)  # adv_batch size : torch.Size([8, 14, 3, 416, 416])
                msk_batch = mypad(msk_batch)  # adv_batch size : torch.Size([8, 14, 3, 416, 416])
                # print("adv_batch size : "+str(adv_batch.size()))

                # Rotation and rescaling transforms
                anglesize = (lab_batch.size(0) * lab_batch.size(1))  # 8*14 = 112
                if do_rotate:
                    angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle,
                                                                       self.maxangle)  # torch.Size([112])
                else:
                    angle = torch.cuda.FloatTensor(anglesize).fill_(0)

                # Resizes and rotates
                current_patch_size = adv_patch.size(-1)
                lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)  # torch.Size([8, 14, 5])
                lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
                lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
                lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
                lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
                target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(scale_rate)) ** 2) + (
                        (lab_batch_scaled[:, :, 4].mul(scale_rate)) ** 2))  # torch.Size([8, 14])
                target_x = lab_batch[:, :, 1].view(np.prod(batch_size))  # torch.Size([112]) 8*14
                target_y = lab_batch[:, :, 2].view(np.prod(batch_size))  # torch.Size([112]) 8*14
                targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))  # torch.Size([112]) 8*14
                targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))  # torch.Size([112]) 8*14
                if (rand_loc):
                    off_x = targetoff_x * (torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))
                    target_x = target_x + off_x
                    off_y = targetoff_y * (torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))
                    target_y = target_y + off_y
                target_y = target_y - 0.05
                # print("current_patch_size : "+str(current_patch_size))
                # print("target_size        : "+str(target_size.size()))
                # print("target_size        : "+str(target_size))
                scale = target_size / current_patch_size  # torch.Size([8, 14])
                scale = scale.view(anglesize)  # torch.Size([112]) 8*14
                # print("scale : "+str(scale))

                s = adv_batch.size()
                adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])  # torch.Size([112, 3, 416, 416])
                msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])  # torch.Size([112, 3, 416, 416])

                tx = (-target_x + 0.5) * 2
                ty = (-target_y + 0.5) * 2
                sin = torch.sin(angle)
                cos = torch.cos(angle)

                # Theta = rotation,rescale matrix
                theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)  # torch.Size([112, 2, 3])
                theta[:, 0, 0] = (cos / scale)
                theta[:, 0, 1] = sin / scale
                theta[:, 0, 2] = (tx * cos / scale + ty * sin / scale)
                theta[:, 1, 0] = -sin / scale
                theta[:, 1, 1] = (cos / scale)
                theta[:, 1, 2] = (-tx * sin / scale + ty * cos / scale)

                if (by_rectangle):
                    theta[:, 1, 1] = theta[:, 1, 1] / 1.5
                    theta[:, 1, 2] = theta[:, 1, 2] / 1.5
                # print(tx)
                # print(theta[:, 0, 2])
                # print(1*cos/scale)
                # print(-1*cos/scale)

                # print("theta :\n"+str(theta))
                # sys.exit()

                b_sh = adv_batch.shape  # b_sh = torch.Size([112, 3, 416, 416])
                grid = F.affine_grid(theta, adv_batch.shape)  # torch.Size([112, 416, 416, 2])

                adv_batch_t = F.grid_sample(adv_batch, grid)  # torch.Size([112, 3, 416, 416])
                msk_batch_t = F.grid_sample(msk_batch, grid)  # torch.Size([112, 3, 416, 416])

                # print("grid : "+str(grid[0,200:300,200:300,:]))

                # msk_batch_t_r = msk_batch_t[:,0,:,:]
                # msk_batch_t_g = msk_batch_t[:,0,:,:]
                # msk_batch_t_b = msk_batch_t[:,0,:,:]
                # for t in range(msk_batch_t.size()[0]):
                #     dx = int(grid[t,0,0,0])
                #     dx2 = int(grid[t,400,400,0])
                #     dy = int(grid[t,0,0,1])
                #     dy2 = int(grid[t,400,400,1])
                #     msk_batch_t[t,0,dx:dx2,dy:dy2] = 0
                #     msk_batch_t[t,1,dx:dx2,dy:dy2] = 0
                #     msk_batch_t[t,2,dx:dx2,dy:dy2] = 0

                # # angle 2
                # tx = (-target_x+0.5)*2
                # ty = (-target_y+0.5)*2
                # sin = torch.sin(angle)
                # cos = torch.cos(angle)

                # # Theta = rotation,rescale matrix
                # theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)  # torch.Size([112, 2, 3])
                # theta[:, 0, 0] = cos/scale
                # theta[:, 0, 1] = sin/scale
                # theta[:, 0, 2] = 0
                # theta[:, 1, 0] = -sin/scale
                # theta[:, 1, 1] = cos/scale
                # theta[:, 1, 2] = 0

                '''
                # Theta2 = translation matrix
                theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
                theta2[:, 0, 0] = 1
                theta2[:, 0, 1] = 0
                theta2[:, 0, 2] = (-target_x + 0.5) * 2
                theta2[:, 1, 0] = 0
                theta2[:, 1, 1] = 1
                theta2[:, 1, 2] = (-target_y + 0.5) * 2

                grid2 = F.affine_grid(theta2, adv_batch.shape)
                adv_batch_t = F.grid_sample(adv_batch_t, grid2)
                msk_batch_t = F.grid_sample(msk_batch_t, grid2)

                '''
                adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])  # torch.Size([8, 14, 3, 416, 416])
                msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])  # torch.Size([8, 14, 3, 416, 416])

                if (with_black_trans):
                    adv_batch_t = torch.clamp(adv_batch_t, 0.0, 0.99999)
                else:
                    adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.99999)
                # img = msk_batch_t[0, 0, :, :, :].detach().cpu()
                # img = transforms.ToPILImage()(img)
                # img.show()
                # exit()

                # output: torch.Size([8, 14, 3, 416, 416]), torch.Size([8, 14, 3, 416, 416])
                # return adv_batch_t * msk_batch_t, (adv_batch_t * msk_batch_t0), (adv_batch_t * msk_batch_t1), (adv_batch_t * msk_batch_t2),  (adv_batch_t * msk_batch_t3), adv_batch_t, msk_batch_t
                return (adv_batch_t * msk_batch_t), msk_batch_t
            elif cls_id_attacked == 11:
                if (with_black_trans):
                    adv_batch = torch.clamp(adv_batch, 0.0, 0.99999)
                else:
                    adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

                # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
                cls_ids = torch.narrow(lab_batch, 2, 0, 1)  # torch.Size([8, 14, 1])
                for cls_id in cls_ids:
                    for item in cls_id:
                        if item[0] == cls_id_attacked:
                            item[0] = 0.
                cls_mask = cls_ids.expand(-1, -1, 3)  # torch.Size([8, 14, 3])
                cls_mask = cls_mask.unsqueeze(-1)  # torch.Size([8, 14, 3, 1])
                cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))  # torch.Size([8, 14, 3, 300])
                cls_mask = cls_mask.unsqueeze(-1)  # torch.Size([8, 14, 3, 300, 1])
                cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))  # torch.Size([8, 14, 3, 300, 300])
                msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(
                    1) - cls_mask  # torch.Size([8, 14, 3, 300, 300])

                # Pad patch and mask to image dimensions
                # Determine size of padding
                pad1 = (img_size - msk_batch.size(-1)) / 2  # (416-300) / 2 = 58
                pad2 = (img_size - msk_batch.size(-2)) / 2  # (416-150) / 2 = 113
                # print("pad : "+str(pad))
                mypad = nn.ConstantPad2d((int(pad1), int(pad1), int(pad2), int(pad2)), 0)
                # print("adv_batch size : "+str(adv_batch.size()))
                adv_batch = mypad(adv_batch)  # adv_batch size : torch.Size([8, 14, 3, 416, 416])
                msk_batch = mypad(msk_batch)  # adv_batch size : torch.Size([8, 14, 3, 416, 416])
                # print("adv_batch size : "+str(adv_batch.size()))

                # Rotation and rescaling transforms
                anglesize = (lab_batch.size(0) * lab_batch.size(1))  # 8*14 = 112
                if do_rotate:
                    angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle,
                                                                       self.maxangle)  # torch.Size([112])
                else:
                    angle = torch.cuda.FloatTensor(anglesize).fill_(0)

                # Resizes and rotates
                current_patch_size = adv_patch.size(-1)
                lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)  # torch.Size([8, 14, 5])
                lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
                lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
                lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
                lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
                target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(scale_rate)) ** 2) + (
                        (lab_batch_scaled[:, :, 4].mul(scale_rate)) ** 2))  # torch.Size([8, 14])
                target_x = lab_batch[:, :, 1].view(np.prod(batch_size))  # torch.Size([112]) 8*14
                target_y = lab_batch[:, :, 2].view(np.prod(batch_size))  # torch.Size([112]) 8*14
                targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))  # torch.Size([112]) 8*14
                targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))  # torch.Size([112]) 8*14
                if (rand_loc):
                    off_x = targetoff_x * (torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))
                    target_x = target_x + off_x
                    off_y = targetoff_y * (torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))
                    target_y = target_y + off_y

                # print("current_patch_size : "+str(current_patch_size))
                # print("target_size        : "+str(target_size.size()))
                # print("target_size        : "+str(target_size))
                scale = target_size / current_patch_size  # torch.Size([8, 14])
                scale = scale.view(anglesize)  # torch.Size([112]) 8*14
                # print("scale : "+str(scale))

                s = adv_batch.size()
                adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])  # torch.Size([112, 3, 416, 416])
                msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])  # torch.Size([112, 3, 416, 416])
                # target_y = target_y - 0.05
                if position == 'bottom':
                    target_y = target_y + 0.5 * targetoff_y * self.bias_coordinate  # 2.25
                elif position == 'top':
                    target_y = target_y - 0.5 * targetoff_y * self.bias_coordinate
                elif position == 'left':
                    target_x = target_x - 0.5 * targetoff_x * self.bias_coordinate
                elif position == 'right':
                    target_x = target_x + 0.5 * targetoff_x * self.bias_coordinate
                tx = (-target_x + 0.5) * 2
                ty = (-target_y + 0.5) * 2
                sin = torch.sin(angle)
                cos = torch.cos(angle)

                # Theta = rotation,rescale matrix
                theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)  # torch.Size([112, 2, 3])
                theta[:, 0, 0] = cos / scale
                theta[:, 0, 1] = sin / scale
                theta[:, 0, 2] = (tx * cos / scale + ty * sin / scale)
                theta[:, 1, 0] = -sin / scale
                theta[:, 1, 1] = cos / scale
                theta[:, 1, 2] = (-tx * sin / scale + ty * cos / scale)

                if (by_rectangle):
                    theta[:, 1, 1] = theta[:, 1, 1] / 1.5
                    theta[:, 1, 2] = theta[:, 1, 2] / 1.5
                # print(tx)
                # print(theta[:, 0, 2])
                # print(1*cos/scale)
                # print(-1*cos/scale)

                # print("theta :\n"+str(theta))
                # sys.exit()

                b_sh = adv_batch.shape  # b_sh = torch.Size([112, 3, 416, 416])
                grid = F.affine_grid(theta, adv_batch.shape)  # torch.Size([112, 416, 416, 2])

                adv_batch_t = F.grid_sample(adv_batch, grid)  # torch.Size([112, 3, 416, 416])
                msk_batch_t = F.grid_sample(msk_batch, grid)  # torch.Size([112, 3, 416, 416])

                # print("grid : "+str(grid[0,200:300,200:300,:]))

                # msk_batch_t_r = msk_batch_t[:,0,:,:]
                # msk_batch_t_g = msk_batch_t[:,0,:,:]
                # msk_batch_t_b = msk_batch_t[:,0,:,:]
                # for t in range(msk_batch_t.size()[0]):
                #     dx = int(grid[t,0,0,0])
                #     dx2 = int(grid[t,400,400,0])
                #     dy = int(grid[t,0,0,1])
                #     dy2 = int(grid[t,400,400,1])
                #     msk_batch_t[t,0,dx:dx2,dy:dy2] = 0
                #     msk_batch_t[t,1,dx:dx2,dy:dy2] = 0
                #     msk_batch_t[t,2,dx:dx2,dy:dy2] = 0

                # # angle 2
                # tx = (-target_x+0.5)*2
                # ty = (-target_y+0.5)*2
                # sin = torch.sin(angle)
                # cos = torch.cos(angle)

                # # Theta = rotation,rescale matrix
                # theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)  # torch.Size([112, 2, 3])
                # theta[:, 0, 0] = cos/scale
                # theta[:, 0, 1] = sin/scale
                # theta[:, 0, 2] = 0
                # theta[:, 1, 0] = -sin/scale
                # theta[:, 1, 1] = cos/scale
                # theta[:, 1, 2] = 0

                '''
                # Theta2 = translation matrix
                theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
                theta2[:, 0, 0] = 1
                theta2[:, 0, 1] = 0
                theta2[:, 0, 2] = (-target_x + 0.5) * 2
                theta2[:, 1, 0] = 0
                theta2[:, 1, 1] = 1
                theta2[:, 1, 2] = (-target_y + 0.5) * 2

                grid2 = F.affine_grid(theta2, adv_batch.shape)
                adv_batch_t = F.grid_sample(adv_batch_t, grid2)
                msk_batch_t = F.grid_sample(msk_batch_t, grid2)

                '''
                adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])  # torch.Size([8, 14, 3, 416, 416])
                msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])  # torch.Size([8, 14, 3, 416, 416])
                if (with_black_trans):
                    adv_batch_t = torch.clamp(adv_batch_t, 0.0, 0.99999)
                else:
                    adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.99999)
                # img = msk_batch_t[0, 0, :, :, :].detach().cpu()
                # img = transforms.ToPILImage()(img)
                # img.show()
                # exit()

                # output: torch.Size([8, 14, 3, 416, 416]), torch.Size([8, 14, 3, 416, 416])
                # return adv_batch_t * msk_batch_t, (adv_batch_t * msk_batch_t0), (adv_batch_t * msk_batch_t1), (adv_batch_t * msk_batch_t2),  (adv_batch_t * msk_batch_t3), adv_batch_t, msk_batch_t
                return (adv_batch_t * msk_batch_t), msk_batch_t

        # adv_batch_masked, adv_batch_masked0, adv_batch_masked1, adv_batch_masked3, adv_batch_masked4, adv_batch_t, msk_batch_t = resize_rotate(adv_batch)
        adv_batch_masked, msk_batch = resize_rotate(adv_batch,
                                                    by_rectangle,
                                                    cls_id_attacked,
                                                    position)  # adv_batch torch.Size([8, 7, 3, 150, 150])   adv_batch_masked torch.Size([8, 7, 3, 416, 416])

        if (with_projection):
            adv_batch = adv_batch_masked
            # # Rotating a Image
            b, f, c, h, w = adv_batch.size()
            adv_batch = adv_batch.view(b * f, c, h, w)
            # print("adv_batch "+str(adv_batch.size())+"  "+str(adv_batch.dtype))
            batch, channel, width, height = adv_batch.size()
            padding_borader = torch.nn.ZeroPad2d(50)
            input_ = padding_borader(adv_batch)
            # print("input_ "+str(input_.size())+"  "+str(input_.dtype))
            angle = np.random.randint(low=-50, high=51)
            mat = get_warpR(anglex=0, angley=angle, anglez=0, fov=42, w=width, h=height)
            mat = mat.expand(batch, -1, -1, -1)
            # print("image  "+str(self.image.dtype)+"  "+str(self.image.size()))
            # print("input_ "+str(input_.dtype)+"  "+str(input_.size()))
            # print("mat    "+str(mat.dtype)+"  "+str(mat.size()))
            adv_batch = tgm.warp_perspective(input_, mat, (input_.size()[-2], input_.size()[-1]))
            # print("adv_batch "+str(adv_batch.size())+"  "+str(adv_batch.dtype))
            adv_batch = adv_batch.view(b, f, c, input_.size()[-2], input_.size()[-1])
            # print("adv_batch "+str(adv_batch.size())+"  "+str(adv_batch.dtype))
            ##
            # Pad patch and mask to image dimensions
            # Determine size of padding
            pad = (img_size - adv_batch.size(-1)) / 2  # (416-300) / 2 = 58
            mypad = nn.ConstantPad2d((int(pad), int(pad), int(pad), int(pad)), 0)
            adv_batch = mypad(adv_batch)  # adv_batch size : torch.Size([8, 14, 3, 416, 416])
            adv_batch_masked = adv_batch

        # adv_batch_masked = torch.clamp(adv_batch_masked, 0.0, 0.99999)
        # return adv_batch_masked, adv_batch_masked0, adv_batch_masked1, adv_batch_masked3, adv_batch_masked4, adv_batch_t, msk_batch_t, adv_patch_set
        return adv_batch_masked, adv_patch_set, msk_batch


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)
        '''
        kernel = torch.cuda.FloatTensor([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,3,-1,-1)
        '''

    def rect_occluding(self, num_rect=1, n_batch=8, n_feature=14, patch_size=300, with_cuda=True):
        if (with_cuda):
            device = 'cuda:0'
        else:
            device = 'cpu'
        tensor_img = torch.full((3, patch_size, patch_size), 0.0).to(device)
        for ttt in range(num_rect):
            xs = torch.randint(0, int(patch_size / 2), (1,))[0]
            xe = torch.randint(xs,
                               torch.min(torch.tensor(tensor_img.size()[-1]), xs + int(patch_size / 2)),
                               (1,))[0]
            ys = torch.randint(0, int(patch_size / 2), (1,))[0]
            ye = torch.randint(ys,
                               torch.min(torch.tensor(tensor_img.size()[-1]), ys + int(patch_size / 2)),
                               (1,))[0]
            tensor_img[:, xs:xe, ys:ye] = 0.5
        tensor_img_batch = tensor_img.unsqueeze(0)  ##  torch.Size([1, 3, 300, 300])
        tensor_img_batch = tensor_img_batch.expand(n_batch, n_feature, -1, -1, -1)  ##  torch.Size([8, 14, 3, 300, 300])
        return tensor_img_batch.to(device)

    def forward(self, adv_patch, lab_batch, img_size, patch_mask=[], cls_id_attacked=11, by_rectangle=False, do_rotate=True, rand_loc=True,
                with_black_trans=False, scale_rate=0.2, with_crease=False, with_projection=False,
                with_rectOccluding=False, enable_empty_patch=False, enable_no_random=False, enable_blurred=True, num_of_round=0):
        try:
            if cls_id_attacked == 0 and num_of_round >0:
                raise Exception
        except Exception as e:
            print("Please choose a appropriate number of patches (like num_of_patches = 1) if you want to attack person.")
            sys.exit(1)

        try:
            if cls_id_attacked == 11 and num_of_round >1:
                raise Exception
        except Exception as e:
            print("Please choose a appropriate number of patches (like num_of_patches = 2) if you want to attack stop sign.")
            sys.exit(1)
        # torch.set_printoptions(edgeitems=sys.maxsize)
        # print("adv_patch size: "+str(adv_patch.size()))
        # patch_size = adv_patch.size(2)

        # init adv_patch. torch.Size([3, 128, 128])
        adv_patch_size = adv_patch.size()[1:3]
        if adv_patch_size[0] > img_size or adv_patch_size[1] > img_size:  # > img_size(416)
            adv_patch = adv_patch.unsqueeze(0)
            if cls_id_attacked == 0:
                adv_patch = F.interpolate(adv_patch, size=img_size)
            if cls_id_attacked == 11:
                adv_patch = F.interpolate(adv_patch, size=(int(img_size/2), img_size))
            adv_patch = adv_patch[0]

        # st()
        # np.save('gg', adv_batch.cpu().detach().numpy())
        # gg=np.load('gg.npy')   np.argwhere(gg!=adv_batch.cpu().detach().numpy())
        def deg_to_rad(deg):
            return torch.tensor(deg * pi / 180.0).float().cuda()

        def rad_to_deg(rad):
            return torch.tensor(rad * 180.0 / pi).float().cuda()

        def get_warpR(anglex, angley, anglez, fov, w, h):
            fov = torch.tensor(fov).float().cuda()
            w = torch.tensor(w).float().cuda()
            h = torch.tensor(h).float().cuda()
            z = torch.sqrt(w ** 2 + h ** 2) / 2 / torch.tan(deg_to_rad(fov / 2)).float().cuda()
            rx = torch.tensor([[1, 0, 0, 0],
                               [0, torch.cos(deg_to_rad(anglex)), -torch.sin(deg_to_rad(anglex)), 0],
                               [0, -torch.sin(deg_to_rad(anglex)), torch.cos(deg_to_rad(anglex)), 0, ],
                               [0, 0, 0, 1]]).float().cuda()
            ry = torch.tensor([[torch.cos(deg_to_rad(angley)), 0, torch.sin(deg_to_rad(angley)), 0],
                               [0, 1, 0, 0],
                               [-torch.sin(deg_to_rad(angley)), 0, torch.cos(deg_to_rad(angley)), 0, ],
                               [0, 0, 0, 1]]).float().cuda()
            rz = torch.tensor([[torch.cos(deg_to_rad(anglez)), torch.sin(deg_to_rad(anglez)), 0, 0],
                               [-torch.sin(deg_to_rad(anglez)), torch.cos(deg_to_rad(anglez)), 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]]).float().cuda()
            r = torch.matmul(torch.matmul(rx, ry), rz)
            pcenter = torch.tensor([h / 2, w / 2, 0, 0]).float().cuda()
            p1 = torch.tensor([0, 0, 0, 0]).float().cuda() - pcenter
            p2 = torch.tensor([w, 0, 0, 0]).float().cuda() - pcenter
            p3 = torch.tensor([0, h, 0, 0]).float().cuda() - pcenter
            p4 = torch.tensor([w, h, 0, 0]).float().cuda() - pcenter
            dst1 = torch.matmul(r, p1)
            dst2 = torch.matmul(r, p2)
            dst3 = torch.matmul(r, p3)
            dst4 = torch.matmul(r, p4)
            list_dst = [dst1, dst2, dst3, dst4]
            org = torch.tensor([[0, 0],
                                [w, 0],
                                [0, h],
                                [w, h]]).float().cuda()
            dst = torch.zeros((4, 2)).float().cuda()
            for i in range(4):
                dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
                dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]
            org = org.unsqueeze(0)
            dst = dst.unsqueeze(0)
            warpR = tgm.get_perspective_transform(org, dst).float().cuda()
            return warpR

        ## get y gray
        # adv_patch_yuv = Colorspace("rgb", "yuv")(adv_patch).cuda()
        # y = adv_patch_yuv[0].unsqueeze(0)
        # adv_patch_new_y_gray = torch.cat((y,y,y), 0).cuda()
        ## get   gray
        # y = (0.2989 * adv_patch[0] + 0.5870 * adv_patch[1] + 0.1140 * adv_patch[2]).unsqueeze(0)
        # adv_patch_new_y_gray = torch.cat((y,y,y), 0).cuda()
        # adv_patch = adv_patch_new_y_gray

        def warping(input_tensor_img, wrinkle_p=15):
            C, H, W = input_tensor_img.size()
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float()  # torch.Size([2, H, W])
            # print("grid "+str(grid.shape)+" : \n"+str(grid))
            grid = grid.view(2, -1)  # torch.Size([2, H*W])
            grid = grid.permute(1, 0)  # torch.Size([H*W, 2])
            perturbed_mesh = grid

            # nv = np.random.randint(20) - 1
            nv = wrinkle_p
            for k in range(nv):
                # Choosing one vertex randomly
                vidx = np.random.randint(grid.shape[0])
                vtex = grid[vidx, :]
                # Vector between all vertices and the selected one
                xv = perturbed_mesh - vtex
                # Random movement 
                mv = (np.random.rand(1, 2) - 0.5) * 20
                hxv = np.zeros((np.shape(xv)[0], np.shape(xv)[1] + 1))
                hxv[:, :-1] = xv
                hmv = np.tile(np.append(mv, 0), (np.shape(xv)[0], 1))
                d = np.cross(hxv, hmv)
                d = np.absolute(d[:, 2])
                # print("d "+str(d.shape)+" :\n"+str(d))
                d = d / (np.linalg.norm(mv, ord=2))
                wt = d
                curve_type = np.random.rand(1)
                if curve_type > 0.3:
                    alpha = np.random.rand(1) * 50 + 50
                    wt = alpha / (wt + alpha)
                else:
                    alpha = np.random.rand(1) + 1
                    wt = 1 - (wt / 100) ** alpha
                msmv = mv * np.expand_dims(wt, axis=1)
                perturbed_mesh = perturbed_mesh + msmv

            perturbed_mesh_2 = perturbed_mesh.permute(1, 0)
            max_x = torch.max(perturbed_mesh_2[0])
            min_x = torch.min(perturbed_mesh_2[0])
            # print("max_x : "+str(max_x)+" / min_x : "+str(min_x))
            max_y = torch.max(perturbed_mesh_2[1])
            min_y = torch.min(perturbed_mesh_2[1])
            # print("max_y : "+str(max_y)+" / min_y : "+str(min_y))
            perturbed_mesh_2[0, :] = (W - 1) * (perturbed_mesh_2[0, :] - min_x) / (max_x - min_x)
            perturbed_mesh_2[1, :] = (H - 1) * (perturbed_mesh_2[1, :] - min_y) / (max_y - min_y)
            # max_x = torch.max(perturbed_mesh_2[0])
            # min_x = torch.min(perturbed_mesh_2[0])
            # print("max_x : "+str(max_x)+" / min_x : "+str(min_x))
            # max_y = torch.max(perturbed_mesh_2[1])
            # min_y = torch.min(perturbed_mesh_2[1])
            # print("max_y : "+str(max_y)+" / min_y : "+str(min_y))
            perturbed_mesh_2 = perturbed_mesh_2.contiguous().view(-1, H, W).float()
            # print("perturbed_mesh_2 dtype : "+str(perturbed_mesh_2.data.type()))
            # print("perturbed_mesh_2 "+str(perturbed_mesh_2.shape)+" : \n"+str(perturbed_mesh_2))

            vgrid = perturbed_mesh_2.unsqueeze(0).cuda()
            vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / (W - 1) - 1.0  # max(W-1,1)
            vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / (H - 1) - 1.0  # max(H-1,1)
            vgrid = vgrid.permute(0, 2, 3, 1).cuda()
            input_tensor_img_b = input_tensor_img.unsqueeze(0).cuda()
            output = F.grid_sample(input_tensor_img_b, vgrid, align_corners=True)  # torch.Size([1, 3, H, W])
            return output[0]

        if (with_crease):
            # warping
            adv_patch = warping(adv_patch)  # torch.Size([3, H, W])
            # print("adv_patch "+str(adv_patch.size())+"  "+str(adv_patch.dtype))

        #
        # adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        if (enable_blurred):
            adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        else:
            adv_patch = adv_patch.unsqueeze(0)
        # print("adv_patch medianpooler size: "+str(adv_patch.size())) ## torch.Size([1, 3, 300, 300])
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)  ##  torch.Size([1, 1, 3, 300, 300])
        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)  ##  torch.Size([8, 14, 3, 300, 300])
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

        if not (len(patch_mask) == 0):
            ## mask size : torch.Size([3, 300, 300])
            patch_mask = patch_mask.unsqueeze(0)  ## mask size : torch.Size([1, 3, 300, 300])
            mask_batch = patch_mask.expand(lab_batch.size(0), lab_batch.size(1), -1, -1,
                                           -1)  ## mask size : torch.Size([8, 14, 3, 300, 300])

        # Contrast, brightness and noise transforms

        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()
        # print("contrast size : "+str(contrast.size()))  ##  contrast size : torch.Size([8, 14, 3, 300, 300])

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()
        # print("brightness size : "+str(brightness.size())) ##  brightness size : torch.Size([8, 14, 3, 300, 300])

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
        # print("noise size : "+str(noise.size()))  ##  noise size : torch.Size([8, 14, 3, 300, 300])
        # print(noise[0,0,0,:10,0])
        # # Apply contrast/brightness/noise, clamp

        # print("adv_patch  : "+str(adv_patch.is_cuda))
        # print("adv_batch  : "+str(adv_batch.is_cuda))
        # print("contrast   : "+str(contrast.is_cuda))
        # print("brightness : "+str(brightness.is_cuda))
        # print("noise      : "+str(noise.is_cuda))

        ## adv_patch 已經模糊
        if enable_no_random and not (enable_empty_patch):
            adv_batch = adv_batch
        if not (enable_no_random) and not (enable_empty_patch):
            adv_batch = adv_batch * contrast + brightness + noise
        if not (len(patch_mask) == 0):
            adv_batch = adv_batch * mask_batch
        if (with_rectOccluding):
            rect_occluder = self.rect_occluding(num_rect=2, n_batch=adv_batch.size()[0], n_feature=adv_batch.size()[1],
                                                patch_size=adv_batch.size()[-1])
            adv_batch = torch.where((rect_occluder == 0), adv_batch, rect_occluder)

        # # get   gray
        # # print("adv_batch size: "+str(adv_batch.size()))  ##  torch.Size([8, 14, 3, 300, 300])
        # # adv_batch = adv_batch.cpu()
        # adv_batch_r = adv_batch[:,:,0,:,:]  ##  torch.Size([3, 300, 300])
        # adv_batch_g = adv_batch[:,:,1,:,:]  ##  torch.Size([3, 300, 300])
        # adv_batch_b = adv_batch[:,:,2,:,:]  ##  torch.Size([3, 300, 300])
        # # print("adv_batch_r size: "+str(adv_batch_r.size()))  ##  torch.Size([8, 14, 300, 300])
        # y = (0.2989 * adv_batch_r + 0.5870 * adv_batch_g + 0.1140 * adv_batch_b)
        # y = y.unsqueeze(2)
        # # print("y size: "+str(y.size()))  ##  torch.Size([8, 14, 3, 300, 300])
        # adv_batch_new_y_gray = torch.cat((y,y,y), 2).cuda()
        # adv_batch = adv_batch_new_y_gray
        # # print("adv_batch size: "+str(adv_batch.size()))  ##  torch.Size([8, 14, 3, 300, 300])
        # # adv_batch = adv_batch.cuda()

        #
        if (with_black_trans):
            adv_batch = torch.clamp(adv_batch, 0.0, 0.99999)
        else:
            adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)
        adv_patch_set = adv_batch[0, 0]

        # ## split img
        # # print("adv_batch size : "+str(adv_batch.size()))  ##  torch.Size([8, 14, 3, 300, 300])
        # adv_batch_units = []
        # split_side = 2
        # split_step = patch_size / split_side
        # # print("split_step : "+str(split_step))
        # for stx in range(0, split_side):
        #     for sty in range(0, split_side):
        #         x_s = int(0 + stx*split_step)
        #         y_s = int(0 + sty*split_step)
        #         x_e = int(x_s+split_step)
        #         y_e = int(y_s+split_step)
        #         adv_batch_unit = adv_batch[:,:,:, x_s:x_e, y_s:y_e].cuda()
        #         adv_batch_zeroes = torch.zeros(adv_batch.size()).cuda()
        #         adv_batch_zeroes[:,:,:, x_s:x_e, y_s:y_e] = adv_batch_unit
        #         adv_batch_unit = adv_batch_zeroes
        #         adv_batch_units.append(adv_batch_unit)

        def resize_rotate(adv_batch, by_rectangle=False, cls_id_attacked = 11, num_of_round = 0):
            if cls_id_attacked == 0:
                if (with_black_trans):
                    adv_batch = torch.clamp(adv_batch, 0.0, 0.99999)
                else:
                    adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

                # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
                cls_ids = torch.narrow(lab_batch, 2, 0, 1)  # torch.Size([8, 14, 1])
                cls_mask = cls_ids.expand(-1, -1, 3)  # torch.Size([8, 14, 3])
                cls_mask = cls_mask.unsqueeze(-1)  # torch.Size([8, 14, 3, 1])
                cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))  # torch.Size([8, 14, 3, 300])
                cls_mask = cls_mask.unsqueeze(-1)  # torch.Size([8, 14, 3, 300, 1])
                cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))  # torch.Size([8, 14, 3, 300, 300])
                msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask  # torch.Size([8, 14, 3, 300, 300])

                # Pad patch and mask to image dimensions
                # Determine size of padding
                pad = (img_size - msk_batch.size(-1)) / 2  # (416-300) / 2 = 58
                # print("pad : "+str(pad))
                mypad = nn.ConstantPad2d((int(pad), int(pad), int(pad), int(pad)), 0)
                # print("adv_batch size : "+str(adv_batch.size()))
                adv_batch = mypad(adv_batch)  # adv_batch size : torch.Size([8, 14, 3, 416, 416])
                msk_batch = mypad(msk_batch)  # adv_batch size : torch.Size([8, 14, 3, 416, 416])
                # print("adv_batch size : "+str(adv_batch.size()))

                # Rotation and rescaling transforms
                anglesize = (lab_batch.size(0) * lab_batch.size(1))  # 8*14 = 112
                if do_rotate:
                    angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)  # torch.Size([112])
                else:
                    angle = torch.cuda.FloatTensor(anglesize).fill_(0)

                # Resizes and rotates
                current_patch_size = adv_patch.size(-1)
                lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)  # torch.Size([8, 14, 5])
                lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
                lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
                lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
                lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
                target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(scale_rate)) ** 2) + (
                            (lab_batch_scaled[:, :, 4].mul(scale_rate)) ** 2))  # torch.Size([8, 14])
                target_x = lab_batch[:, :, 1].view(np.prod(batch_size))  # torch.Size([112]) 8*14
                target_y = lab_batch[:, :, 2].view(np.prod(batch_size))  # torch.Size([112]) 8*14
                targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))  # torch.Size([112]) 8*14
                targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))  # torch.Size([112]) 8*14
                if (rand_loc):
                    off_x = targetoff_x * (torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))
                    target_x = target_x + off_x
                    off_y = targetoff_y * (torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))
                    target_y = target_y + off_y
                target_y = target_y - 0.05
                # print("current_patch_size : "+str(current_patch_size))
                # print("target_size        : "+str(target_size.size()))
                # print("target_size        : "+str(target_size))
                scale = target_size / current_patch_size  # torch.Size([8, 14])
                scale = scale.view(anglesize)  # torch.Size([112]) 8*14
                # print("scale : "+str(scale))

                s = adv_batch.size()
                adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])  # torch.Size([112, 3, 416, 416])
                msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])  # torch.Size([112, 3, 416, 416])

                tx = (-target_x + 0.5) * 2
                ty = (-target_y + 0.5) * 2
                sin = torch.sin(angle)
                cos = torch.cos(angle)

                # Theta = rotation,rescale matrix
                theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)  # torch.Size([112, 2, 3])
                theta[:, 0, 0] = (cos / scale)
                theta[:, 0, 1] = sin / scale
                theta[:, 0, 2] = (tx * cos / scale + ty * sin / scale)
                theta[:, 1, 0] = -sin / scale
                theta[:, 1, 1] = (cos / scale)
                theta[:, 1, 2] = (-tx * sin / scale + ty * cos / scale)

                if (by_rectangle):
                    theta[:, 1, 1] = theta[:, 1, 1] / 1.5
                    theta[:, 1, 2] = theta[:, 1, 2] / 1.5
                # print(tx)
                # print(theta[:, 0, 2])
                # print(1*cos/scale)
                # print(-1*cos/scale)

                # print("theta :\n"+str(theta))
                # sys.exit()

                b_sh = adv_batch.shape  # b_sh = torch.Size([112, 3, 416, 416])
                grid = F.affine_grid(theta, adv_batch.shape)  # torch.Size([112, 416, 416, 2])

                adv_batch_t = F.grid_sample(adv_batch, grid)  # torch.Size([112, 3, 416, 416])
                msk_batch_t = F.grid_sample(msk_batch, grid)  # torch.Size([112, 3, 416, 416])

                # print("grid : "+str(grid[0,200:300,200:300,:]))

                # msk_batch_t_r = msk_batch_t[:,0,:,:]
                # msk_batch_t_g = msk_batch_t[:,0,:,:]
                # msk_batch_t_b = msk_batch_t[:,0,:,:]
                # for t in range(msk_batch_t.size()[0]):
                #     dx = int(grid[t,0,0,0])
                #     dx2 = int(grid[t,400,400,0])
                #     dy = int(grid[t,0,0,1])
                #     dy2 = int(grid[t,400,400,1])
                #     msk_batch_t[t,0,dx:dx2,dy:dy2] = 0
                #     msk_batch_t[t,1,dx:dx2,dy:dy2] = 0
                #     msk_batch_t[t,2,dx:dx2,dy:dy2] = 0

                # # angle 2
                # tx = (-target_x+0.5)*2
                # ty = (-target_y+0.5)*2
                # sin = torch.sin(angle)
                # cos = torch.cos(angle)

                # # Theta = rotation,rescale matrix
                # theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)  # torch.Size([112, 2, 3])
                # theta[:, 0, 0] = cos/scale
                # theta[:, 0, 1] = sin/scale
                # theta[:, 0, 2] = 0
                # theta[:, 1, 0] = -sin/scale
                # theta[:, 1, 1] = cos/scale
                # theta[:, 1, 2] = 0

                '''
                # Theta2 = translation matrix
                theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
                theta2[:, 0, 0] = 1
                theta2[:, 0, 1] = 0
                theta2[:, 0, 2] = (-target_x + 0.5) * 2
                theta2[:, 1, 0] = 0
                theta2[:, 1, 1] = 1
                theta2[:, 1, 2] = (-target_y + 0.5) * 2
        
                grid2 = F.affine_grid(theta2, adv_batch.shape)
                adv_batch_t = F.grid_sample(adv_batch_t, grid2)
                msk_batch_t = F.grid_sample(msk_batch_t, grid2)
        
                '''
                adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])  # torch.Size([8, 14, 3, 416, 416])
                msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])  # torch.Size([8, 14, 3, 416, 416])

                if (with_black_trans):
                    adv_batch_t = torch.clamp(adv_batch_t, 0.0, 0.99999)
                else:
                    adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.99999)
                # img = msk_batch_t[0, 0, :, :, :].detach().cpu()
                # img = transforms.ToPILImage()(img)
                # img.show()
                # exit()

                # output: torch.Size([8, 14, 3, 416, 416]), torch.Size([8, 14, 3, 416, 416])
                # return adv_batch_t * msk_batch_t, (adv_batch_t * msk_batch_t0), (adv_batch_t * msk_batch_t1), (adv_batch_t * msk_batch_t2),  (adv_batch_t * msk_batch_t3), adv_batch_t, msk_batch_t
                return (adv_batch_t * msk_batch_t), msk_batch_t
            elif cls_id_attacked == 11:
                if (with_black_trans):
                    adv_batch = torch.clamp(adv_batch, 0.0, 0.99999)
                else:
                    adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

                # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
                cls_ids = torch.narrow(lab_batch, 2, 0, 1)  # torch.Size([8, 14, 1])
                for item in cls_ids:
                    if item[0] == cls_id_attacked:
                        item[0] = 0.
                cls_mask = cls_ids.expand(-1, -1, 3)  # torch.Size([8, 14, 3])
                cls_mask = cls_mask.unsqueeze(-1)  # torch.Size([8, 14, 3, 1])
                cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))  # torch.Size([8, 14, 3, 300])
                cls_mask = cls_mask.unsqueeze(-1)  # torch.Size([8, 14, 3, 300, 1])
                cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))  # torch.Size([8, 14, 3, 300, 300])
                msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(
                    1) - cls_mask  # torch.Size([8, 14, 3, 300, 300])

                # Pad patch and mask to image dimensions
                # Determine size of padding
                pad1 = (img_size - msk_batch.size(-1)) / 2  # (416-300) / 2 = 58
                pad2 = (img_size - msk_batch.size(-2)) / 2  # (416-150) / 2 = 113
                # print("pad : "+str(pad))
                mypad = nn.ConstantPad2d((int(pad1), int(pad1), int(pad2), int(pad2)), 0)
                # print("adv_batch size : "+str(adv_batch.size()))
                adv_batch = mypad(adv_batch)  # adv_batch size : torch.Size([8, 14, 3, 416, 416])
                msk_batch = mypad(msk_batch)  # adv_batch size : torch.Size([8, 14, 3, 416, 416])
                # print("adv_batch size : "+str(adv_batch.size()))

                # Rotation and rescaling transforms
                anglesize = (lab_batch.size(0) * lab_batch.size(1))  # 8*14 = 112
                if do_rotate:
                    angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle,
                                                                       self.maxangle)  # torch.Size([112])
                else:
                    angle = torch.cuda.FloatTensor(anglesize).fill_(0)

                # Resizes and rotates
                current_patch_size = adv_patch.size(-1)
                lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)  # torch.Size([8, 14, 5])
                lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
                lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
                lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
                lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
                target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(scale_rate)) ** 2) + (
                        (lab_batch_scaled[:, :, 4].mul(scale_rate)) ** 2))  # torch.Size([8, 14])
                target_x = lab_batch[:, :, 1].view(np.prod(batch_size))  # torch.Size([112]) 8*14
                target_y = lab_batch[:, :, 2].view(np.prod(batch_size))  # torch.Size([112]) 8*14
                targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))  # torch.Size([112]) 8*14
                targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))  # torch.Size([112]) 8*14
                if (rand_loc):
                    off_x = targetoff_x * (torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))
                    target_x = target_x + off_x
                    off_y = targetoff_y * (torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))
                    target_y = target_y + off_y

                # print("current_patch_size : "+str(current_patch_size))
                # print("target_size        : "+str(target_size.size()))
                # print("target_size        : "+str(target_size))
                scale = target_size / current_patch_size  # torch.Size([8, 14])
                scale = scale.view(anglesize)  # torch.Size([112]) 8*14
                # print("scale : "+str(scale))

                s = adv_batch.size()
                adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])  # torch.Size([112, 3, 416, 416])
                msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])  # torch.Size([112, 3, 416, 416])
                # target_y = target_y - 0.05
                if num_of_round == 0:
                    target_y = target_y - 0.5 * targetoff_y * 3 / 5
                elif num_of_round == 1:
                    target_y = target_y + 0.5 * targetoff_y * 3 / 5
                tx = (-target_x + 0.5) * 2
                ty = (-target_y + 0.5) * 2
                sin = torch.sin(angle)
                cos = torch.cos(angle)

                # Theta = rotation,rescale matrix
                theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)  # torch.Size([112, 2, 3])
                theta[:, 0, 0] = cos / scale
                theta[:, 0, 1] = sin / scale
                theta[:, 0, 2] = (tx * cos / scale + ty * sin / scale)
                theta[:, 1, 0] = -sin / scale
                theta[:, 1, 1] = cos / scale
                theta[:, 1, 2] = (-tx * sin / scale + ty * cos / scale)

                if (by_rectangle):
                    theta[:, 1, 1] = theta[:, 1, 1] / 1.5
                    theta[:, 1, 2] = theta[:, 1, 2] / 1.5
                # print(tx)
                # print(theta[:, 0, 2])
                # print(1*cos/scale)
                # print(-1*cos/scale)

                # print("theta :\n"+str(theta))
                # sys.exit()

                b_sh = adv_batch.shape  # b_sh = torch.Size([112, 3, 416, 416])
                grid = F.affine_grid(theta, adv_batch.shape)  # torch.Size([112, 416, 416, 2])

                adv_batch_t = F.grid_sample(adv_batch, grid)  # torch.Size([112, 3, 416, 416])
                msk_batch_t = F.grid_sample(msk_batch, grid)  # torch.Size([112, 3, 416, 416])

                # print("grid : "+str(grid[0,200:300,200:300,:]))

                # msk_batch_t_r = msk_batch_t[:,0,:,:]
                # msk_batch_t_g = msk_batch_t[:,0,:,:]
                # msk_batch_t_b = msk_batch_t[:,0,:,:]
                # for t in range(msk_batch_t.size()[0]):
                #     dx = int(grid[t,0,0,0])
                #     dx2 = int(grid[t,400,400,0])
                #     dy = int(grid[t,0,0,1])
                #     dy2 = int(grid[t,400,400,1])
                #     msk_batch_t[t,0,dx:dx2,dy:dy2] = 0
                #     msk_batch_t[t,1,dx:dx2,dy:dy2] = 0
                #     msk_batch_t[t,2,dx:dx2,dy:dy2] = 0

                # # angle 2
                # tx = (-target_x+0.5)*2
                # ty = (-target_y+0.5)*2
                # sin = torch.sin(angle)
                # cos = torch.cos(angle)

                # # Theta = rotation,rescale matrix
                # theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)  # torch.Size([112, 2, 3])
                # theta[:, 0, 0] = cos/scale
                # theta[:, 0, 1] = sin/scale
                # theta[:, 0, 2] = 0
                # theta[:, 1, 0] = -sin/scale
                # theta[:, 1, 1] = cos/scale
                # theta[:, 1, 2] = 0

                '''
                # Theta2 = translation matrix
                theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
                theta2[:, 0, 0] = 1
                theta2[:, 0, 1] = 0
                theta2[:, 0, 2] = (-target_x + 0.5) * 2
                theta2[:, 1, 0] = 0
                theta2[:, 1, 1] = 1
                theta2[:, 1, 2] = (-target_y + 0.5) * 2

                grid2 = F.affine_grid(theta2, adv_batch.shape)
                adv_batch_t = F.grid_sample(adv_batch_t, grid2)
                msk_batch_t = F.grid_sample(msk_batch_t, grid2)

                '''
                adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])  # torch.Size([8, 14, 3, 416, 416])
                msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])  # torch.Size([8, 14, 3, 416, 416])
                if (with_black_trans):
                    adv_batch_t = torch.clamp(adv_batch_t, 0.0, 0.99999)
                else:
                    adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.99999)
                # img = msk_batch_t[0, 0, :, :, :].detach().cpu()
                # img = transforms.ToPILImage()(img)
                # img.show()
                # exit()

                # output: torch.Size([8, 14, 3, 416, 416]), torch.Size([8, 14, 3, 416, 416])
                # return adv_batch_t * msk_batch_t, (adv_batch_t * msk_batch_t0), (adv_batch_t * msk_batch_t1), (adv_batch_t * msk_batch_t2),  (adv_batch_t * msk_batch_t3), adv_batch_t, msk_batch_t
                return (adv_batch_t * msk_batch_t), msk_batch_t

        # adv_batch_masked, adv_batch_masked0, adv_batch_masked1, adv_batch_masked3, adv_batch_masked4, adv_batch_t, msk_batch_t = resize_rotate(adv_batch)
        adv_batch_masked, msk_batch = resize_rotate(adv_batch,
                                                    by_rectangle,
                                                    cls_id_attacked,
                                                    num_of_round=num_of_round)  # adv_batch torch.Size([8, 7, 3, 150, 150])   adv_batch_masked torch.Size([8, 7, 3, 416, 416])

        if (with_projection):
            adv_batch = adv_batch_masked
            # # Rotating a Image
            b, f, c, h, w = adv_batch.size()
            adv_batch = adv_batch.view(b * f, c, h, w)
            # print("adv_batch "+str(adv_batch.size())+"  "+str(adv_batch.dtype))
            batch, channel, width, height = adv_batch.size()
            padding_borader = torch.nn.ZeroPad2d(50)
            input_ = padding_borader(adv_batch)
            # print("input_ "+str(input_.size())+"  "+str(input_.dtype))
            angle = np.random.randint(low=-50, high=51)
            mat = get_warpR(anglex=0, angley=angle, anglez=0, fov=42, w=width, h=height)
            mat = mat.expand(batch, -1, -1, -1)
            # print("image  "+str(self.image.dtype)+"  "+str(self.image.size()))
            # print("input_ "+str(input_.dtype)+"  "+str(input_.size()))
            # print("mat    "+str(mat.dtype)+"  "+str(mat.size()))
            adv_batch = tgm.warp_perspective(input_, mat, (input_.size()[-2], input_.size()[-1]))
            # print("adv_batch "+str(adv_batch.size())+"  "+str(adv_batch.dtype))
            adv_batch = adv_batch.view(b, f, c, input_.size()[-2], input_.size()[-1])
            # print("adv_batch "+str(adv_batch.size())+"  "+str(adv_batch.dtype))
            ##
            # Pad patch and mask to image dimensions
            # Determine size of padding
            pad = (img_size - adv_batch.size(-1)) / 2  # (416-300) / 2 = 58
            mypad = nn.ConstantPad2d((int(pad), int(pad), int(pad), int(pad)), 0)
            adv_batch = mypad(adv_batch)  # adv_batch size : torch.Size([8, 14, 3, 416, 416])
            adv_batch_masked = adv_batch

        # adv_batch_masked = torch.clamp(adv_batch_masked, 0.0, 0.99999)
        # return adv_batch_masked, adv_batch_masked0, adv_batch_masked1, adv_batch_masked3, adv_batch_masked4, adv_batch_t, msk_batch_t, adv_patch_set
        return adv_batch_masked, adv_patch_set, msk_batch


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        # print("img_batch size : "+str(img_batch.size()))  ##  torch.Size([8, 3, 416, 416])
        # print("adv_batch size : "+str(adv_batch.size()))  ##  torch.Size([8, 14, 3, 416, 416])
        advs = torch.unbind(adv_batch, 1)
        # print("advs (np) size : "+str(np.array(advs).shape))  ##  (14,)
        # print("b[0].size      : "+str(b[0].size()))  ##  torch.Size([8, 3, 416, 416])
        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)
        return img_batch


'''
class PatchGenerator(nn.Module):
    """PatchGenerator: network module that generates adversarial patches.

    Module representing the neural network that will generate adversarial patches.

    """

    def __init__(self, cfgfile, weightfile, img_dir, lab_dir):
        super(PatchGenerator, self).__init__()
        self.yolo = Darknet(cfgfile).load_weights(weightfile)
        self.dataloader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, shuffle=True),
                                                      batch_size=5,
                                                      shuffle=True)
        self.patchapplier = PatchApplier()
        self.nmscalculator = NMSCalculator()
        self.totalvariation = TotalVariation()

    def forward(self, *input):
        pass
'''


class AdvDataset(Dataset):
    """
    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the dataset.
        lab_dir: Directory containing the labels of the dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.
    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = max_lab

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')
        if os.path.getsize(lab_path):  # check to see if label file contains data.
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([5])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)
        transform = transforms.ToTensor()
        image = transform(image)
        label = self.pad_lab(label)
        return image, label

    def pad_and_scale(self, img, lab):
        """

        Args:
            img:

        Returns:

        """
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
        resize = transforms.Resize((self.imgsize, self.imgsize))
        padded_img = resize(padded_img)  # choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if (pad_size > 0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
        else:
            padded_lab = lab
        return padded_lab

'''
class InriaDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = max_lab

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')
        if os.path.getsize(lab_path):  # check to see if label file contains data.
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([5])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)
        transform = transforms.ToTensor()
        image = transform(image)
        label = self.pad_lab(label)
        return image, label

    def pad_and_scale(self, img, lab):
        """

        Args:
            img:

        Returns:

        """
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
        resize = transforms.Resize((self.imgsize, self.imgsize))
        padded_img = resize(padded_img)  # choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if (pad_size > 0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
        else:
            padded_lab = lab
        return padded_lab
'''

if __name__ == '__main__':
    if len(sys.argv) == 3:
        img_dir = sys.argv[1]
        lab_dir = sys.argv[2]

    else:
        print('Usage: ')
        print('  python load_data.py img_dir lab_dir')
        sys.exit()

    test_loader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, shuffle=True),
                                              batch_size=3, shuffle=True)

    cfgfile = "cfg/yolov2.cfg"
    weightfile = "weights/yolov2.weights"
    printfile = "non_printability/30values.txt"

    patch_size = 400

    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.cuda()
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()
    prob_extractor = MaxProbExtractor(0, 80).cuda()
    nms_calculator = NMSCalculator(printfile, patch_size)
    total_variation = TotalVariation()
    '''
    img = Image.open('data/horse.jpg').convert('RGB')
    img = img.resize((darknet_model.width, darknet_model.height))
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    img = torch.autograd.Variable(img)

    output = darknet_model(img)
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    tl0 = time.time()
    tl1 = time.time()
    for i_batch, (img_batch, lab_batch) in enumerate(test_loader):
        tl1 = time.time()
        print('time to fetch items: ', tl1 - tl0)
        img_batch = img_batch.cuda()
        lab_batch = lab_batch.cuda()
        adv_patch = Image.open('data/horse.jpg').convert('RGB')
        adv_patch = adv_patch.resize((patch_size, patch_size))
        transform = transforms.ToTensor()
        adv_patch = transform(adv_patch).cuda()
        img_size = img_batch.size(-1)
        print('transforming patches')
        t0 = time.time()
        adv_batch_t = patch_transformer.forward(adv_patch, lab_batch, img_size)
        print('applying patches')
        t1 = time.time()
        img_batch = patch_applier.forward(img_batch, adv_batch_t)
        img_batch = torch.autograd.Variable(img_batch)
        img_batch = F.interpolate(img_batch, (darknet_model.height, darknet_model.width))
        print('running patched images through model')
        t2 = time.time()

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    try:
                        print(type(obj), obj.size())
                    except:
                        pass
            except:
                pass

        print(torch.cuda.memory_allocated())

        output = darknet_model(img_batch)
        print('extracting max probs')
        t3 = time.time()
        max_prob = prob_extractor(output)
        t4 = time.time()
        nms = nms_calculator.forward(adv_patch)
        tv = total_variation(adv_patch)
        print('---------------------------------')
        print('        patch transformation : %f' % (t1 - t0))
        print('           patch application : %f' % (t2 - t1))
        print('             darknet forward : %f' % (t3 - t2))
        print('      probability extraction : %f' % (t4 - t3))
        print('---------------------------------')
        print('          total forward pass : %f' % (t4 - t0))
        del img_batch, lab_batch, adv_patch, adv_batch_t, output, max_prob
        torch.cuda.empty_cache()
        tl0 = time.time()
