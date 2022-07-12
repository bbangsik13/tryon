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

        ground_truth_parse_paths, agnostic_mask_paths, warped_cloth_img_paths,warped_cloth_mask_paths, densepose_label_paths, ground_truth_img_paths = self.get_paths(opt)

        util.natural_sort(ground_truth_parse_paths)
        util.natural_sort(agnostic_mask_paths)
        util.natural_sort(warped_cloth_img_paths)
        util.natural_sort(warped_cloth_mask_paths)
        util.natural_sort(densepose_label_paths)
        util.natural_sort(ground_truth_img_paths)

        self.paths_match(ground_truth_img_paths,ground_truth_parse_paths)
        self.paths_match(ground_truth_img_paths,agnostic_mask_paths)
        self.paths_match(ground_truth_img_paths,warped_cloth_img_paths)
        self.paths_match(ground_truth_img_paths,warped_cloth_mask_paths)
        self.paths_match(ground_truth_img_paths,densepose_label_paths)

        self.ground_truth_parse_paths = ground_truth_parse_paths
        self.agnostic_mask_paths = agnostic_mask_paths
        self.warped_cloth_img_paths = warped_cloth_img_paths
        self.warped_cloth_mask_paths = warped_cloth_mask_paths
        self.densepose_label_paths = densepose_label_paths
        self.ground_truth_img_paths = ground_truth_img_paths

        size = len(self.ground_truth_img_paths)

        self.dataset_size = size

    def get_paths(self, opt):
        ground_truth_parse_paths = glob(os.path.join(opt.dataroot,'ground_truth_parse','*'))
        agnostic_mask_paths = glob(os.path.join(opt.dataroot,'agnostic_mask','*'))
        warped_cloth_img_paths = glob(os.path.join(opt.dataroot,'warped_cloth_img','*'))
        warped_cloth_mask_paths = glob(os.path.join(opt.dataroot,'warped_cloth_mask','*'))
        densepose_label_paths = glob(os.path.join(opt.dataroot,'densepose_label','*'))
        ground_truth_img_paths = glob(os.path.join(opt.dataroot,'ground_truth_img','*'))
        #assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return ground_truth_parse_paths, agnostic_mask_paths, warped_cloth_img_paths,warped_cloth_mask_paths, densepose_label_paths, ground_truth_img_paths

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
        # ground truth img
        ground_truth_img_path = self.ground_truth_img_paths[index]
        ground_truth_img = Image.open(ground_truth_img_path)
        ground_truth_img = ground_truth_img.convert('RGB')
        params = get_params(self.opt, ground_truth_img.size)
        transform_img = get_transform(self.opt, params, method=Image.NEAREST, normalize=True, toTensor=True, mask=False)
        ground_truth_img_tensor = transform_img(ground_truth_img)


        # ground_truth_parse
        ground_truth_parse_path = self.ground_truth_parse_paths[index]
        agnostic_img = Image.open(ground_truth_parse_path)
        agnostic_img = agnostic_img.convert('L')
        agnostic_img = np.array(agnostic_img)

        # agnostic mask
        agnostic_mask_path = self.agnostic_mask_paths[index]
        assert self.path_match(ground_truth_img_path, agnostic_mask_path), \
            "The label_path %s and image_path %s don't match." % \
            (ground_truth_img_path, agnostic_mask_path)
        agnostic_mask = Image.open(agnostic_mask_path)
        agnostic_mask = agnostic_mask.convert('L')
        transform_mask = get_transform(self.opt, params, method=Image.NEAREST, normalize=True,toTensor=True,mask=True)
        agnostic_mask_tensor = transform_mask(agnostic_mask)
        agnostic_mask_tensor[agnostic_mask_tensor>0]=1
        agnostic_mask_tensor[agnostic_mask_tensor<0]=0

        # warped cloth image
        warped_cloth_img_path = self.warped_cloth_img_paths[index]
        assert self.path_match(ground_truth_img_path, warped_cloth_img_path), \
            "The label_path %s and image_path %s don't match." % \
            (ground_truth_img_path, warped_cloth_img_path)
        warped_cloth_img = Image.open(warped_cloth_img_path)
        warped_cloth_img = warped_cloth_img.convert('RGB')
        warped_cloth_img_tensor = transform_img(warped_cloth_img)

        # warped cloth mask
        warped_cloth_mask_path = self.warped_cloth_mask_paths[index]
        assert self.path_match(ground_truth_img_path, warped_cloth_mask_path), \
            "The label_path %s and image_path %s don't match." % \
            (ground_truth_img_path, warped_cloth_mask_path)
        warped_cloth_mask = Image.open(warped_cloth_mask_path)
        warped_cloth_mask = warped_cloth_mask.convert('L')
        warped_cloth_mask_tensor = transform_mask(warped_cloth_mask)
        warped_cloth_mask_tensor[warped_cloth_mask_tensor>0]=1
        warped_cloth_mask_tensor[warped_cloth_mask_tensor<0]=0
        # densepose label
        densepose_label_path = self.densepose_label_paths[index]
        assert self.path_match(ground_truth_img_path, densepose_label_path), \
            "The label_path %s and image_path %s don't match." % \
            (ground_truth_img_path, densepose_label_path)
        densepose_label = (np.load(densepose_label_path)[:,:,0]).astype(np.uint8)



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

