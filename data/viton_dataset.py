"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import torch

class VitonDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        agnostic_img_paths, agnostic_mask_paths, warped_cloth_img_paths,warped_cloth_mask_paths, densepose_label_paths, ground_truth_img_paths = self.get_paths(opt)

        util.natural_sort(agnostic_img_paths)
        util.natural_sort(agnostic_mask_paths)
        util.natural_sort(warped_cloth_img_paths)
        util.natural_sort(warped_cloth_mask_paths)
        util.natural_sort(densepose_label_paths)
        util.natural_sort(ground_truth_img_paths)

        self.paths_match(ground_truth_img_paths,agnostic_img_paths)
        self.paths_match(ground_truth_img_paths,agnostic_mask_paths)
        self.paths_match(ground_truth_img_paths,warped_cloth_img_paths)
        self.paths_match(ground_truth_img_paths,warped_cloth_mask_paths)
        self.paths_match(ground_truth_img_paths,densepose_label_paths)

        self.agnostic_img_paths = agnostic_img_paths
        self.agnostic_mask_paths = agnostic_mask_paths
        self.warped_cloth_img_paths = warped_cloth_img_paths
        self.warped_cloth_mask_paths = warped_cloth_mask_paths
        self.densepose_label_paths = densepose_label_paths
        self.ground_truth_img_paths = ground_truth_img_paths

        size = len(self.ground_truth_img_paths)

        self.dataset_size = size

    def get_paths(self, opt):
        agnostic_img_paths = glob(os.path.join(opt.dataroot,'agnostic_img','*'))
        agnostic_mask_paths = glob(os.path.join(opt.dataroot,'agnostic_mask','*'))
        warped_cloth_img_paths = glob(os.path.join(opt.dataroot,'warped_cloth_img','*'))
        warped_cloth_mask_paths = glob(os.path.join(opt.dataroot,'warped_cloth_mask','*'))
        densepose_label_paths = glob(os.path.join(opt.dataroot,'densepose_label','*'))
        ground_truth_img_paths = glob(os.path.join(opt.dataroot,'ground_truth_img','*'))
        #assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return agnostic_img_paths, agnostic_mask_paths, warped_cloth_img_paths,warped_cloth_mask_paths, densepose_label_paths, ground_truth_img_paths

    def path_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def paths_match(self,paths1,paths2):
        if len(paths1) != len(paths2):
            raise ("the data length mismatch")
        for path1, path2 in zip(paths1, paths2):
            assert self.path_match(path1, path2), \
                "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. " \
                "Are you sure about the pairing? and use --no_pairing_check to bypass this." % (path1, path2)

    def __getitem__(self, index):

        ###############################################load data########################################################
        # agnostic image
        agnostic_img_path = self.agnostic_img_paths[index]
        agnostic_img = Image.open(agnostic_img_path)
        agnostic_img = agnostic_img.convert('RGB')
        params = get_params(self.opt, agnostic_img.size)
        transform_img = get_transform(self.opt, params, method=Image.NEAREST, normalize=True,toTensor=True,mask=False)
        agnostic_img_tensor = transform_img(agnostic_img)

        # agnostic mask
        agnostic_mask_path = self.agnostic_mask_paths[index]
        assert self.path_match(agnostic_img_path, agnostic_mask_path), \
            "The label_path %s and image_path %s don't match." % \
            (agnostic_img_path, agnostic_mask_path)
        agnostic_mask = Image.open(agnostic_mask_path)
        agnostic_mask = agnostic_mask.convert('L')
        transform_mask = get_transform(self.opt, params, method=Image.NEAREST, normalize=True,toTensor=True,mask=True)
        agnostic_mask_tensor = transform_mask(agnostic_mask)
        agnostic_mask_tensor[agnostic_mask_tensor>0]=1
        agnostic_mask_tensor[agnostic_mask_tensor<0]=0

        # warped cloth image
        warped_cloth_img_path = self.warped_cloth_img_paths[index]
        assert self.path_match(agnostic_img_path, warped_cloth_img_path), \
            "The label_path %s and image_path %s don't match." % \
            (agnostic_img_path, warped_cloth_img_path)
        warped_cloth_img = Image.open(warped_cloth_img_path)
        warped_cloth_img = warped_cloth_img.convert('RGB')
        warped_cloth_img_tensor = transform_img(warped_cloth_img)

        # warped cloth mask
        warped_cloth_mask_path = self.warped_cloth_mask_paths[index]
        assert self.path_match(agnostic_img_path, warped_cloth_mask_path), \
            "The label_path %s and image_path %s don't match." % \
            (agnostic_img_path, warped_cloth_mask_path)
        warped_cloth_mask = Image.open(warped_cloth_mask_path)
        warped_cloth_mask = warped_cloth_mask.convert('L')
        warped_cloth_mask_tensor = transform_mask(warped_cloth_mask)
        warped_cloth_mask_tensor[warped_cloth_mask_tensor>0]=1
        warped_cloth_mask_tensor[warped_cloth_mask_tensor<0]=0
        # densepose label
        densepose_label_path = self.densepose_label_paths[index]
        assert self.path_match(agnostic_img_path, densepose_label_path), \
            "The label_path %s and image_path %s don't match." % \
            (agnostic_img_path, densepose_label_path)
        densepose_label = (np.load(densepose_label_path)[:,:,0]).astype(np.uint8)


        # ground truth img
        ground_truth_img_path = self.ground_truth_img_paths[index]
        assert self.path_match(agnostic_img_path, ground_truth_img_path), \
            "The label_path %s and image_path %s don't match." % \
            (agnostic_img_path, ground_truth_img_path)
        ground_truth_img = Image.open(ground_truth_img_path)
        ground_truth_img = ground_truth_img.convert('RGB')
        ground_truth_img_tensor = transform_img(ground_truth_img)

        ###############################################convert data#####################################################

        # inpaint mask
        inpaint_mask_tensor = (agnostic_mask_tensor - agnostic_mask_tensor*warped_cloth_mask_tensor)
        inpaint_mask_tensor[inpaint_mask_tensor>0]=1
        inpaint_mask_tensor[inpaint_mask_tensor<0]=0

        # inpaint img
        inpaint_img_tensor = (1-agnostic_mask_tensor)*agnostic_img_tensor + agnostic_mask_tensor * warped_cloth_mask_tensor*warped_cloth_img_tensor
        #inpaint_img_swap_tensor= (1-inpaint_mask_tensor) * ground_truth_img_tensor
        #plt.imshow(densepose_label),plt.show()
        # inpaint parse
        '''
        densepose label information
        0: background
        1: Torso back
        2: Torso front
        3: right hand
        4: left hand
        5: right foot
        6: left foot
        7: upper leg right back
        8: upper leg left back
        9: upper leg right front
        10: upper leg left front
        11: lower leg right back
        12: lower leg left back
        13: lower leg right front
        14: lower leg left front
        15: upper arm left front
        16: upper arm right front
        17: upper arm left back
        18: upper arm right back
        19: lower arm left front
        20: lower arm right front
        21: lower arm left back
        22: lower arm right back
        23: head right
        24: head left
        '''
        new_densepose_label = np.zeros((densepose_label.shape[0],densepose_label.shape[1],5),dtype=np.float32)
        label_map ={
            "background":[0],
            "arm": [15,17,19,21,4,16,18,20,22,3],
            "leg": [8,10,12,14,6,7,9,11,13,5],
            "neck": [23,24],
            "torso": [1,2],
        }
        for i, key in enumerate(label_map.keys()):
            for l in label_map[key]:
                new_densepose_label[:,:,i] += densepose_label == l

        new_densepose_label = np.transpose(new_densepose_label,(2,0,1))
        densepose_label_tensor = torch.from_numpy(new_densepose_label)

        inpaint_parse_tensor = inpaint_mask_tensor * densepose_label_tensor

        '''plt.subplot(2,2,1),plt.imshow(((np.transpose(inpaint_img_tensor.numpy(),(1,2,0))+1)/2*255).astype(np.uint8)),plt.title('inpaint img')
        plt.subplot(2,2,2),plt.imshow(np.transpose(inpaint_mask_tensor.numpy(),(1,2,0))),plt.title('inpaint mask')
        plt.subplot(2,2,3),plt.imshow(tensor2label(densepose_label_tensor,5)),plt.title('parse')
        plt.subplot(2,2,4),plt.imshow(tensor2label(inpaint_parse_tensor,5)),plt.title('inpaint parse')
        plt.show()'''

        input_dict = {'inpaint mask': inpaint_mask_tensor,
                      'inpaint img': inpaint_img_tensor,
                      'inpaint parse': inpaint_parse_tensor,
                      'ground truth img':ground_truth_img_tensor,
                      'path': ground_truth_img_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)],
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
