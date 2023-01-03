'''
Author: Dylan8527 vvm8933@gmail.com
Date: 2022-12-03 15:12:42
LastEditors: Dylan8527 vvm8933@gmail.com
LastEditTime: 2022-12-17 22:37:22
FilePath: \code\dataset.py
Description: 

Copyright (c) 2022 by Dylan8527 vvm8933@gmail.com, All Rights Reserved. 
'''
import utils
import numpy as np
import SimpleITK as sitk
from torch.utils import data

class TrainData(data.Dataset):
    def __init__(self, proj_path, proj_pos_path, num_sample_ray, num_angle, SOD):
        self.num_angle = num_angle # number of projection angle
        self.num_sample_ray = num_sample_ray # number of sample ray per iter i.e. ray[i:i+num_sample_ray]
        self.SOD = SOD # source to object distance
        self.angles = np.linspace(0., 360., num=self.num_angle, endpoint=False)  # (num_angle, )
        self.proj_pos = sitk.GetArrayFromImage(sitk.ReadImage(proj_pos_path)).reshape(-1) # (num_det, )
        self.num_det = len(self.proj_pos)
        # projection & metal_trace
        self.proj = sitk.GetArrayFromImage(sitk.ReadImage(proj_path))  # (num_angle, num_det)
        self.proj /= self.proj.max() # normalize
        # ray
        self.rays = utils.fan_beam_ray(self.proj_pos, self.SOD) # (num_det, 2*SOD, 2)
        self.index_max = self.num_det - self.num_sample_ray
        
    def __getitem__(self, item):
        ang = self.angles[item]
        proj = self.proj[item].reshape(-1, )  # (num_det, )
        # sample ray, projection, and metal trace
        index = np.random.randint(0, self.index_max, size=1)[0]
        ray_sample = self.rays[index:index+self.num_sample_ray]     # (num_sample_ray, 2*SOD, 2)
        proj_sample = proj[index:index+self.num_sample_ray]     # *(num_sample_ray, )
        # rotate ray
        ray_sample = utils.rotate_ray(xy=ray_sample, angle=ang)
        return ray_sample, proj_sample

    def __len__(self):
        return self.num_angle


class TestData(data.Dataset):
    def __init__(self, h, w):
        self.h, self.w = h, w
        self.xy = utils.grid_coordinate(h=self.h, w=self.w).reshape(1, int(h*w), 2)

    def __getitem__(self, item):
        return self.xy[item]    # (h*w, 2)

    def __len__(self):
        return 1
