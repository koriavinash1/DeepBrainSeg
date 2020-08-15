#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# author: Avinash Kori
# contact: koriavinash1@gmail.com
# MIT License

# Copyright (c) 2020 Avinash Kori

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.feature import canny
import skimage.morphology as morph

import torch
from torch.utils.data import Dataset

from ..helpers import preprocessing
from ..brainmask import get_brain_mask

ants_path = os.path.join('/opt/ANTs/bin/')


def nii_loader(paths):
    """
    Now given a path, we have to load
    1) High res version of flair,t2,t1 and t1c
    2) Low  res version of flair,t2,t1 and t1c
    3) If the path comprises of "Non" then make an empty 9x9x9 array
    4) if not, load the segmentation mask too.


    Accepts
    1) path of data

    Returns
    1) a 4D high resolution data
    2) a 4d low resolution data
    3) Segmentation
    """

    flair = nib.load(paths['flair']).get_data()
    t2 = nib.load(paths['t2']).get_data()
    t1 = nib.load(paths['t1']).get_data()
    t1ce = nib.load(paths['t1ce']).get_data()
    affine = nib.load(paths['t1']).affine


    try:
        brain_mask = nib.load(paths['mask']).get_data()
    except:
        brain_mask = get_brain_mask(paths['t1'], ants_path)

    try:
        seg_mask = np.uint8(nib.load(paths['seg']).get_data())
        seg_mask[(brain_mask != 0)*(sege_mask <= 0)] = 5
        seg_mask[np.where(sege_mask==4)] = 3
        seg_mask[np.where(sege_mask==5)] = 4  ## making an effort to make classes 0,1,2,3,4 rather than 0,1,2,4,5
    except:
        seg_mask = None

    t1    = preprocessing.clip(t1)
    t1ce  = preprocessing.clip(t1ce)
    t2    = preprocessing.clip(t2)
    flair = preprocessing.clip(flair)

    t1    = preprocessing.normalize(t1,    brain_mask)
    t1ce  = preprocessing.normalize(t1ce,   brain_mask)
    t2    = preprocessing.normalize(t2,    brain_mask)
    flair = preprocessing.normalize(flair, brain_mask)

    data = {}
    data['flair'] = flair
    data['t2'] = t2
    data['t1'] = t1
    data['t1ce'] = t1ce
    return data, seg_mask, affine


def get_patch(vol, seg = None, coordinate = (0,0,0), size = 64):
    data = np.zeros((4, size, size, size))

    data[0,:,:,:] = vol['flair'][coordinate[0]:coordinate[0] + size,
 	                               coordinate[1]:coordinate[1] + size,
	                               coordinate[2]:coordinate[2] + size]
    data[1,:,:,:] = vol['t2'][coordinate[0]:coordinate[0] + size,
	                               coordinate[1]:coordinate[1] + size,
	                               coordinate[2]:coordinate[2] + size]
    data[2,:,:,:] = vol['t1'][coordinate[0]:coordinate[0] + size,
	                               coordinate[1]:coordinate[1] + size,
	                               coordinate[2]:coordinate[2] + size]
    data[3,:,:,:] = vol['t1ce'][coordinate[0]:coordinate[0] + size,
	                               coordinate[1]:coordinate[1] + size,
	                               coordinate[2]:coordinate[2] + size]
    try:
        seg_mask = seg[coordinate[0]:coordinate[0] + size,
	                               coordinate[1]:coordinate[1] + size,
	                               coordinate[2]:coordinate[2] + size]
        return data, seg_mask
    except:
        return data


def multilabel_binarize(image_nD, nlabel):
    """
    """
    labels = range(nlabel)
    out_shape = (len(labels),) + image_nD.shape
    bin_img_stack = np.ones(out_shape, dtype='uint8')
    for label in labels:
        bin_img_stack[label] = np.where(image_nD == label, bin_img_stack[label], 0)
    return bin_img_stack



selem = morph.disk(1)
def getEdge(label):
    """
    """
    out = np.float32(np.zeros(label.shape))
    for i in range(label.shape[0]):
        out[i,:,:] = morph.binary_dilation(canny(np.float32(label[i,:,:])), selem)
    
    return out


def getEdgeEnhancedWeightMap_3D(label, label_ids =[0,1,2,3,4], scale=1, edgescale=1, assign_equal_wt=False):
    """
    """
    label = multilabel_binarize(label, len(label_ids))# convert to onehot vector
    shape = (0,)+label.shape[1:]
    weight_map = np.empty(shape, dtype='uint8')
    if assign_equal_wt:
        return np.ones_like(label)

    for i in range(label.shape[0]): 
        #Estimate weight maps:
        weights = np.ones(len(label_ids))
        slice_map = np.ones(label[i,:,:,:].shape)
        for _id in label_ids:
            class_frequency = np.sum(label[i,:,:,:] == label_ids[_id])
            if class_frequency:
                weights[label_ids.index(_id)] = scale*label[i,:,:,:].size/class_frequency
                slice_map[np.where(label[i,:,:,:]==label_ids.index(_id))] = weights[label_ids.index(_id)]
                edge = getEdge(label[i,:,:,:])
                edge_frequency = np.sum(np.sum(edge==1.0))
                if edge_frequency:    
                    slice_map[np.where(edge==1.0)] += edgescale*label[i,:,:,:].size/edge_frequency
            # utils.imshow(edge, cmap='gray')
        # utils.imshow(weight_map, cmap='gray')
        weight_map = np.append(weight_map, np.expand_dims(slice_map, axis=0), axis=0)
    return np.sum(np.float32(weight_map), 0)



class Generator(Dataset):
    """
    """

    def __init__(self, csv_path, 
                 patch_size = 64,
                 hardmine_every = 10,
                 batch_size = 64,
                 iteration = 1,
                 loader = nii_loader,
                 patch_extractor = get_patch):

        self.nchannels = 4
        self.nclasses = 5
        self.loader = loader
        self.patch_extractor = patch_extractor
        self.csv = pd.read_csv(csv_path)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.valid_patches = self.csv[self.csv['brain'] > 0]
        self.ratio = iteration*1./hardmine_every	
        self.classinfo = {
            'ET':{
                'hardmine_ratio': self.ratio,
                'hardmine_threshold': 1.0 - self.ratio,
                'subjects': self.valid_patches[self.valid_patches['ETRegion'] > 0],
                'hardsubjects': self.valid_patches[(self.valid_patches['ETRegion'] > 0) * (self.valid_patches['ETdice'] < 1.0 - self.ratio)]
            },
            'TC' :{
                'hardmine_ratio': self.ratio,
                'hardmine_threshold': 1.0 - self.ratio,
                'subjects': self.valid_patches[self.valid_patches['TCRegion'] > 0],
                'hardsubjects': self.valid_patches[(self.valid_patches['TCRegion'] > 0) * (self.valid_patches['TCdice'] < 1.0 - self.ratio)]
            },
            'WT' :{
                'hardmine_ratio': self.ratio,
                'hardmine_threshold': 1.0 - self.ratio,
                'subjects': self.valid_patches[self.valid_patches['WTRegion'] > 0],
                'hardsubjects': self.valid_patches[(self.valid_patches['WTRegion'] > 0) * (self.valid_patches['WTdice'] < 1.0 - self.ratio)]
            },
            'Brain': {
                'hardmine_ratio': self.ratio,
                'hardmine_threshold': 1.0 - self.ratio,
                'subjects': self.valid_patches,
                'hardsubjects': self.valid_patches[(self.valid_patches['brain'] < 1.0 - self.ratio)]
            }
        }

    def __len__(self):
        return len(self.valid_patches)//self.batch_size


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        X, y, Emap = self.__data_generation(index)
        return X, y, Emap

  
    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        self.image_size = (self.patch_size, self.patch_size, self.patch_size)
        X = []; y = []; edgeMap = []


        p = np.random.uniform()

        s = len(self.classinfo.keys())
        for bi in range(self.batch_size):

            for i, key in enumerate(self.classinfo.keys()):
                if (bi%len(self.classinfo)) == i:
                    if p < self.classinfo[key]['hardmine_ratio']:
                        subject = self.classinfo[key]['hardsubjects'].iloc[int(index*self.batch_size//s + bi) % len(self.classinfo[key]['hardsubjects'])]
                    subject = self.classinfo[key]['subjects'].iloc[int(index*self.batch_size//s + bi) % len(self.classinfo[key]['subjects'])]
                    break


            spath = {}; subject_idx = subject['path'].split('/')[-1]; subject_path = subject['path']
            spath['flair'] = os.path.join(subject_path, subject_idx + '_flair.nii.gz')
            spath['t1ce']  = os.path.join(subject_path, subject_idx + '_t1ce.nii.gz')
            spath['seg']   = os.path.join(subject_path, subject_idx + '_seg.nii.gz')
            spath['t1']    = os.path.join(subject_path, subject_idx + '_t1.nii.gz')
            spath['t2']    = os.path.join(subject_path, subject_idx + '_t2.nii.gz')

            spath['mask']  = os.path.join(subject_path, 'mask.nii.gz')
            coordinate = [int(co) for co in subject['coordinate'][1:-1].split(', ')]

            vol, seg, _ = self.loader(spath)
            data, mask = self.path_extractor(vol, seg, 
                                        coordinate = coordinate, 
                                        size = self.patch_size)

            X.append(data)
            y.append(mask)
            # edgeMap.append(getEdgeEnhancedWeightMap_3D(mask))

        return X, y, edgeMap


if __name__ == '__main__':
    csv_path = '../../../../Logs/csv/training.csv'
    datasetTrain = Generator(csv_path = csv_path,
                                                batch_size = 8,
                                                hardmine_every = 10,
                                                iteration = 1)
    from torch.utils.data import DataLoader
    dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=1, shuffle=True,  num_workers=1, pin_memory=True)
    for i, (x,y,w) in enumerate(dataLoaderTrain):
        y = torch.cat(y).long().squeeze(0)
        x = torch.cat(x).float().squeeze(0)
        # w = torch.cat(w).float().squeeze(0) / torch.max(w)
        print(i, x.shape, y.shape) #, w.shape)
