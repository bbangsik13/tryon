"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os
import os.path as osp
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import torch
import json
import cv2
from matplotlib import pyplot as plt

class VitonDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        parser.add_argument('--bottom_agnostic', action='store_true',help='use when bottom tryon')
        parser.add_argument('--augmentation', action='store_true',help='data augmentation')

        return parser

    def initialize(self, opt):
        self.opt = opt

        ground_truth_parse_dir = osp.join(
            opt.dataroot, 'ground_truth_parse')
        warped_cloth_img_dir = osp.join(
            opt.dataroot, 'warped_cloth_img')
        warped_cloth_mask_dir = osp.join(
            opt.dataroot, 'warped_cloth_mask')
        densepose_label_dir = osp.join(
            opt.dataroot, 'densepose_label')
        openpose_json_dir = osp.join(
            opt.dataroot, 'openpose_json')
        ground_truth_img_dir = osp.join(
            opt.dataroot, 'ground_truth_img')

        self.ground_truth_parse_dir = ground_truth_parse_dir
        self.warped_cloth_img_dir = warped_cloth_img_dir
        self.warped_cloth_mask_dir = warped_cloth_mask_dir
        self.densepose_label_dir = densepose_label_dir
        self.openpose_json_dir = openpose_json_dir
        self.ground_truth_img_dir = ground_truth_img_dir

        data_ids = [path.split('/')[-1].split('_')[0] for path in glob(osp.join(warped_cloth_img_dir, '*'))]

        self.data_ids = data_ids

        self.dataset_size = len(data_ids)

        self.bottom_agnostic = opt.bottom_agnostic



    def __getitem__(self, index):

        ###############################################load data########################################################

        model_id = '1657518529204' # self.data_ids[index]
        cloth_id = self.data_ids[index]

        # ground truth img
        ground_truth_img_path = osp.join(
            self.ground_truth_img_dir, model_id + '_model.jpg')
        ground_truth_img = Image.open(ground_truth_img_path)
        ground_truth_img = ground_truth_img.convert('RGB')
        params = get_params(self.opt, ground_truth_img.size)
        transform_img = get_transform(
            self.opt, params, method=Image.NEAREST, normalize=True, toTensor=True, mask=False)
        ground_truth_img_tensor = transform_img(ground_truth_img)
        # ground_truth_parse
        ground_truth_parse_path = osp.join(
            self.ground_truth_parse_dir, model_id + '_model.png')
        ground_truth_parse = Image.open(ground_truth_parse_path)
        ground_truth_parse = ground_truth_parse
        ground_truth_parse = np.array(ground_truth_parse)

        # openpose_json
        openpose_json_path = osp.join(
            self.openpose_json_dir, model_id + '_model_keypoints.json')
        with open(openpose_json_path, 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]

        part_arrays = self.get_part_arrays(ground_truth_parse, pose_data)
        agnostic_mask = self.get_agnostic_mask(
            part_arrays, bottom_agnostic=self.bottom_agnostic)
        if self.opt.bottom_agnostic:
            agnostic_mask = agnostic_mask[0]+agnostic_mask[1]+agnostic_mask[2]
        else:
            agnostic_mask = agnostic_mask[0]+agnostic_mask[1]+agnostic_mask[2]+agnostic_mask[3]

        agnostic_mask_tensor = torch.from_numpy(agnostic_mask[np.newaxis,:,:])
        agnostic_mask_tensor[agnostic_mask_tensor > 0] = 1
        agnostic_mask_tensor[agnostic_mask_tensor < 0] = 0

        cloth_type = 'btm' if self.bottom_agnostic else 'top'

        # warped cloth image

        warped_cloth_img_path = osp.join(
            self.warped_cloth_img_dir, cloth_id + '_' + cloth_type + '.png')

        warped_cloth_img = Image.open(warped_cloth_img_path)
        warped_cloth_img = warped_cloth_img.convert('RGB')
        warped_cloth_img_tensor = transform_img(warped_cloth_img)

        # warped cloth mask
        warped_cloth_mask_path = osp.join(
            self.warped_cloth_mask_dir, cloth_id + '_' + cloth_type + '.png')
        warped_cloth_mask = Image.open(warped_cloth_mask_path)
        warped_cloth_mask = np.array(warped_cloth_mask.convert('L'))

        warped_cloth_mask_tensor = torch.from_numpy(warped_cloth_mask[np.newaxis,:,:])  # FIXME
        warped_cloth_mask_tensor[warped_cloth_mask_tensor > 0] = 1
        warped_cloth_mask_tensor[warped_cloth_mask_tensor < 0] = 0
        # densepose label
        densepose_label_path = osp.join(
            self.densepose_label_dir, model_id + '_model.npy')
        densepose_label = (np.load(densepose_label_path)[:, :, 0]).astype(np.uint8)

        ###############################################convert data#####################################################
        '''
        0:background
        1:hair
        2:none
        3:none
        4:face+head
        5:top
        6:none
        7:outer
        8:none
        9:bottom
        10:neck
        11:none
        12:none
        13:none
        14:left arm,left hand
        15:right arm, right hand
        16:left leg, left foot
        17:right leg, right foot
        18:none
        19:none
        20:etc(torso skin)
        21:hat
        22:bag
        23:glove
        '''
        new_ground_truth_parse = np.zeros(
            (ground_truth_parse.shape[0], ground_truth_parse.shape[1], 9), dtype=np.float32)
        parse_label_map = {
            "background": [0], # only background
            "arm": [14,15], # integrate arms
            "leg": [16,17], # integrate legs
            "neck": [4,10], # neck & face
            "torso skin": [20], # belly
            "top":[5], # top cloth
            "bottom":[9], # bottom cloth
            "outer":[7], # outter
            "etc":[1,2,3,6,8,11,12,13,18,19,21,22,23] # accessory & hair & not defined label
        }
        for i, key in enumerate(parse_label_map.keys()):
            for l in parse_label_map[key]:
                new_ground_truth_parse[:, :, i] += ground_truth_parse == l

        new_ground_truth_parse = np.transpose(new_ground_truth_parse, (2, 0, 1))
        new_ground_truth_parse_tensor = torch.from_numpy(new_ground_truth_parse)

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
        new_densepose_label = np.zeros(
            (densepose_label.shape[0], densepose_label.shape[1], 5), dtype=np.float32)
        label_map = {
            "background": [0],
            "arm": [15, 17, 19, 21, 4, 16, 18, 20, 22, 3],
            "leg": [8, 10, 12, 14, 6, 7, 9, 11, 13, 5],
            "neck": [23, 24],
            "torso skin": [1, 2],
        }
        for i, key in enumerate(label_map.keys()):
            for l in label_map[key]:
                new_densepose_label[:, :, i] += densepose_label == l

        new_densepose_label = np.transpose(new_densepose_label, (2, 0, 1))
        new_densepose_label_tensor = torch.from_numpy(new_densepose_label)

        if self.opt.augmentation:# data augmentation-agnostic_mask
            agnostic_mask = cv2.dilate(agnostic_mask,np.ones((3,3)),iterations=np.random.randint(14)+1)
            if cloth_type == "top":
                agnostic_mask = agnostic_mask * (new_ground_truth_parse[0]+new_ground_truth_parse[1]
                                                +((ground_truth_parse==10)*1).astype(np.float32) +new_ground_truth_parse[5])
            else:
                agnostic_mask = agnostic_mask * (new_ground_truth_parse[0] + new_ground_truth_parse[2]
                                                 + new_ground_truth_parse[6])
            agnostic_mask_tensor = torch.from_numpy(agnostic_mask[np.newaxis, :, :])
            agnostic_mask_tensor[agnostic_mask_tensor > 0] = 1
            agnostic_mask_tensor[agnostic_mask_tensor < 0] = 0

        # inpaint mask
        inpaint_mask_tensor = (agnostic_mask_tensor - warped_cloth_mask_tensor)
        inpaint_mask_tensor[inpaint_mask_tensor > 0] = 1
        inpaint_mask_tensor[inpaint_mask_tensor < 0] = 0

        if cloth_type == "top":
            #swap_mask_tensor = (agnostic_mask_tensor - agnostic_mask_tensor * new_ground_truth_parse_tensor[5] * warped_cloth_mask_tensor)
            swap_mask_tensor = (agnostic_mask_tensor - agnostic_mask_tensor * new_ground_truth_parse_tensor[5])
        else:
            #swap_mask_tensor = (agnostic_mask_tensor - agnostic_mask_tensor * new_ground_truth_parse_tensor[6] * warped_cloth_mask_tensor)
            swap_mask_tensor = (agnostic_mask_tensor - agnostic_mask_tensor * new_ground_truth_parse_tensor[6])
        swap_mask_tensor[swap_mask_tensor > 0] = 1
        swap_mask_tensor[swap_mask_tensor < 0] = 0
        #plt.imshow(np.transpose(((swap_mask_tensor.numpy()+1)/2*255).astype(np.uint8),(1,2,0))),plt.title('swap_mask_tensor'),plt.show()


        # inpaint img
        inpaint_img_tensor = (1-agnostic_mask_tensor)*(1-warped_cloth_mask_tensor)*ground_truth_img_tensor + \
            warped_cloth_mask_tensor * warped_cloth_img_tensor
        #plt.imshow(np.transpose(((inpaint_img_tensor.numpy()+1)/2*255).astype(np.uint8),(1,2,0))),plt.title('inpaint img'),plt.show()

        if cloth_type == "top":
            #swap_img_tensor = (1 - agnostic_mask_tensor) * ground_truth_img_tensor + \
            #                 agnostic_mask_tensor * warped_cloth_mask_tensor *new_ground_truth_parse_tensor[5] * ground_truth_img_tensor
            swap_img_tensor = (1 - agnostic_mask_tensor) * ground_truth_img_tensor + \
                              agnostic_mask_tensor * new_ground_truth_parse_tensor[5] * ground_truth_img_tensor
        else:
            #swap_img_tensor = (1 - agnostic_mask_tensor) * ground_truth_img_tensor + \
            #                  agnostic_mask_tensor * warped_cloth_mask_tensor * new_ground_truth_parse_tensor[
            #                      6] * ground_truth_img_tensor
            swap_img_tensor = (1 - agnostic_mask_tensor) * ground_truth_img_tensor + \
                              agnostic_mask_tensor * new_ground_truth_parse_tensor[6] * ground_truth_img_tensor
        #plt.imshow(np.transpose(((swap_img_tensor.numpy() + 1) / 2 * 255).astype(np.uint8), (1, 2, 0))), plt.title('swap img'), plt.show()
        # inpaint parse
        inpaint_parse_tensor = new_ground_truth_parse_tensor*(1-agnostic_mask_tensor)*(1-warped_cloth_mask_tensor)
        for i in range(5):
            inpaint_parse_tensor[i:i+1,:,:] += new_densepose_label_tensor[i:i+1,:,:]*inpaint_mask_tensor
        if cloth_type == "top":
            inpaint_parse_tensor[5:6,:,:] += warped_cloth_mask_tensor
        else:
            inpaint_parse_tensor[6:7,:,:] += warped_cloth_mask_tensor

        swap_parse_tensor = new_ground_truth_parse_tensor * (1 - agnostic_mask_tensor)
        for i in range(5):
            swap_parse_tensor[i:i+1,:,:] += new_densepose_label_tensor[i:i+1,:,:] * swap_mask_tensor
        if cloth_type == "top":
            swap_parse_tensor[5:6,:,:] += new_ground_truth_parse_tensor[5:6,:,:] * agnostic_mask_tensor
        else:
            swap_parse_tensor[6:7,:,:] += new_ground_truth_parse_tensor[6:7,:,:] * agnostic_mask_tensor
        '''if cloth_type == "top":
            #swap_parse_tensor = torch.cat((new_ground_truth_parse_tensor * (1 - agnostic_mask_tensor),
            #                              warped_cloth_mask_tensor * agnostic_mask_tensor * new_ground_truth_parse_tensor[5],
            #                              new_densepose_label_tensor * swap_mask_tensor), dim=0)
            swap_parse_tensor = torch.cat((new_ground_truth_parse_tensor * (1 - agnostic_mask_tensor),
                                           agnostic_mask_tensor * new_ground_truth_parse_tensor[5],
                                           new_densepose_label_tensor * swap_mask_tensor), dim=0)
        else:
            #swap_parse_tensor = torch.cat((new_ground_truth_parse_tensor * (1 - agnostic_mask_tensor),
            #                               warped_cloth_mask_tensor * agnostic_mask_tensor *
            #                               new_ground_truth_parse_tensor[6],
            #                               new_densepose_label_tensor * swap_mask_tensor), dim=0)
            swap_parse_tensor = torch.cat((new_ground_truth_parse_tensor * (1 - agnostic_mask_tensor),
                                           agnostic_mask_tensor * new_ground_truth_parse_tensor[6],
                                           new_densepose_label_tensor * swap_mask_tensor), dim=0)
'''
        #print(swap_img_tensor[:,56,43])
        #print(swap_img_tensor[:,401,468])
        '''plt.subplot(2,3,1),plt.imshow(util.tensor2label(new_ground_truth_parse_tensor, 8)),plt.title('new ground truth parse')
        plt.subplot(2,3,2),plt.imshow(util.tensor2label(new_densepose_label_tensor, 8)),plt.title('new densepose label')
        plt.subplot(2,3,3),plt.imshow(np.transpose(agnostic_mask_tensor.numpy(),(1,2,0))),plt.title('agnostic mask')
        plt.subplot(2,3,4),plt.imshow(util.tensor2label(swap_parse_tensor, swap_parse_tensor.shape[0])),plt.title('swap parse')
        plt.subplot(2,3,5),plt.imshow(util.tensor2label(inpaint_parse_tensor, inpaint_parse_tensor.shape[0])),plt.title('inpaint parse')
        plt.subplot(2,3,6),plt.imshow(((np.transpose(swap_img_tensor.numpy(),(1,2,0))+1)/2*255).astype(np.uint8)),plt.title('warped_cloth')
        plt.show()'''
        #plt.imshow(util.tensor2label(swap_parse_tensor, swap_parse_tensor.shape[0])),plt.show()


        '''plt.subplot(2,2,1),plt.imshow(((np.transpose(inpaint_img_tensor.numpy(),(1,2,0))+1)/2*255).astype(np.uint8)),plt.title('inpaint img')
        plt.subplot(2,2,2),plt.imshow(np.transpose(inpaint_mask_tensor.numpy(),(1,2,0))),plt.title('inpaint mask')
        plt.subplot(2,2,3),plt.imshow(tensor2label(densepose_label_tensor,5)),plt.title('parse')
        plt.subplot(2,2,4),plt.imshow(tensor2label(inpaint_parse_tensor,5)),plt.title('inpaint parse')
        plt.show()'''
        input_dict = {'inpaint mask': inpaint_mask_tensor,
                      'inpaint img': inpaint_img_tensor,
                      'inpaint parse': inpaint_parse_tensor,
                      'swap mask': swap_mask_tensor,
                      'swap img': swap_img_tensor,
                      'swap parse': swap_parse_tensor,
                      'ground truth img': ground_truth_img_tensor,
                      'path': warped_cloth_img_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size

    def get_part_arrays(self, parse, pose_data):
        ''' common processing for Ia and Sa
            separate feet from legs for bottom
            separate hands from arms for upper
        '''

        # 1. separate each labels  @TODO why float?
        parse_array = np.array(parse)
        parse_background = (parse_array == 0).astype(np.float32)
        parse_head = ((parse_array == 4).astype(np.float32) +
                      (parse_array == 13).astype(np.float32))
        parse_neck = (parse_array == 10).astype(np.float32)  # Check it is OK
        parse_l_arm = (parse_array == 14).astype(np.float32)
        parse_r_arm = (parse_array == 15).astype(np.float32)
        parse_l_leg = (parse_array == 16).astype(np.float32)
        parse_r_leg = (parse_array == 17).astype(np.float32)
        parse_upper = ((parse_array == 5).astype(np.float32) +
                       (parse_array == 6).astype(np.float32) +
                       (parse_array == 7).astype(np.float32))
        parse_lower = ((parse_array == 9).astype(np.float32) +
                       (parse_array == 12).astype(np.float32) +
                       (parse_array == 18).astype(np.float32) +
                       (parse_array == 19).astype(np.float32))

        # @TODO Do we need both foot and arm ?

        # get parse_hands and arms
        # if not bottom_agnostic
        parse_hands = []
        parse_arms = []
        for parse_id, pose_ids in [(14, [6, 7]), (15, [3, 4])]:
            pt_elbow = pose_data[pose_ids[0]]
            pt_wrist = pose_data[pose_ids[1]]
            vec_arm = (pt_wrist - pt_elbow)
            len_arm = np.linalg.norm(pt_wrist - pt_elbow)
            # @TODO  not 0 but quite small (what if invisible case?)
            if len_arm != 0:
                vec_arm /= len_arm
            vec_cut_arm = vec_arm[::-1] * np.array([1, -1])
            if np.sum(parse_array == parse_id) == 0:  # @TODO ???
                parse_arms.append(np.zeros_like(parse_array))
                parse_hands.append(np.zeros_like(parse_array))
                continue
            parse_arm = 255 * (parse_array == parse_id).astype(np.uint8)
            pt1 = tuple((pt_wrist - 10000 * vec_cut_arm / 2).astype(np.int32))
            pt2 = tuple((pt_wrist + 10000 * vec_cut_arm / 2).astype(np.int32))
            cv2.line(parse_arm, pt1, pt2, color=0, thickness=1)
            # print("parse_arm.shape:", parse_arm.shape)
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
                parse_arm, connectivity=4)
            for x, y in zip(np.arange(pt_elbow[0], pt_wrist[0] + 2e-8, vec_arm[0] + 1e-8), np.arange(pt_elbow[1], pt_wrist[1] + 2e-8, vec_arm[1] + 1e-8)):
                label_arm = labels[int(y), int(x)]
                if label_arm != 0:
                    break
            if label_arm == 0:
                label_arm = -1
            parse_arm = (labels == label_arm).astype(np.float32)
            parse_hand = (parse_array == parse_id).astype(
                np.float32) - parse_arm
            parse_arms.append(parse_arm)
            parse_hands.append(parse_hand)

        parse_l_arm, parse_r_arm = parse_arms    # seperate left & right
        parse_l_hand, parse_r_hand = parse_hands  # separate left & right

        # get parse_foot and legs
        # if self.bottom_agnostic
        parse_foot = []
        parse_legs = []
        for parse_id, pose_ids in [(16, [13, 14]), (17, [10, 11])]:
            pt_knee = pose_data[pose_ids[0]]
            pt_ankle = pose_data[pose_ids[1]]
            vec_leg = (pt_ankle - pt_knee)
            len_leg = np.linalg.norm(pt_ankle - pt_knee)
            if len_leg != 0:
                vec_leg /= len_leg
            vec_cut_leg = vec_leg[::-1] * np.array([1, -1])
            if np.sum(parse_array == parse_id) == 0:
                parse_legs.append(np.zeros_like(parse_array))
                parse_foot.append(np.zeros_like(parse_array))
                continue
            parse_leg = 255 * (parse_array == parse_id).astype(np.uint8)
            pt1 = tuple((pt_ankle - 10000 * vec_cut_leg / 2).astype(np.int32))
            pt2 = tuple((pt_ankle + 10000 * vec_cut_leg / 2).astype(np.int32))
            cv2.line(parse_leg, pt1, pt2, color=0, thickness=1)
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
                parse_leg, connectivity=4)
            for x, y in zip(np.arange(pt_knee[0], pt_ankle[0] + 2e-8, vec_leg[0] + 1e-8), np.arange(pt_knee[1], pt_ankle[1] + 2e-8, vec_leg[1] + 1e-8)):
                label_leg = labels[int(y), int(x)]
                if label_leg != 0:
                    break
            if label_leg == 0:
                label_leg = -1
            parse_leg = (labels == label_leg).astype(np.float32)
            parse_feet = (parse_array == parse_id).astype(
                np.float32) - parse_leg
            parse_legs.append(parse_leg)
            parse_foot.append(parse_feet)

        parse_l_leg, parse_r_leg = parse_legs   # separate left & right leg
        parse_l_feet, parse_r_feet = parse_foot  # separate left & right foot

        part_arrays = {
            'parse_background': parse_background,
            'parse_head': parse_head,
            'parse_neck': parse_neck,
            'parse_l_arm': parse_l_arm,
            'parse_r_arm': parse_r_arm,
            'parse_l_hand': parse_l_hand,
            'parse_r_hand': parse_r_hand,
            'parse_l_leg': parse_l_leg,
            'parse_r_leg': parse_r_leg,
            'parse_l_feet': parse_l_feet,
            'parse_r_feet': parse_r_feet,
            'parse_upper': parse_upper,
            'parse_lower': parse_lower
        }

        return part_arrays

    def get_agnostic_mask(self, part_arrays, bottom_agnostic):
        '''
            merge mask regions
            @TODO get a single array?
        '''
        if bottom_agnostic:  # lower
            parse_lower = part_arrays['parse_lower']
            parse_l_leg = part_arrays['parse_l_leg']
            parse_r_leg = part_arrays['parse_r_leg']
            mask_parts = [parse_l_leg, parse_r_leg, parse_lower]
        else:  # upper
            parse_upper = part_arrays['parse_upper']
            parse_neck = part_arrays['parse_neck']
            parse_l_arm = part_arrays['parse_l_arm']
            parse_r_arm = part_arrays['parse_r_arm']
            mask_parts = [parse_neck, parse_l_arm, parse_r_arm, parse_upper]

        return mask_parts
