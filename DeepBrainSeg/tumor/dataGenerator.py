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


def nii_loader(paths):
    """
        Loads entire patient data i.e volumes with all 
        the modalities provided 

        args
            paths: json with ['flair', 't1', 't2', 't1ce'] keys,
                with optional keys of ['mask', 'seg']
        returns
            data: json with ['flair', 't1', 't2', 't1ce'] keys,
                and their respective data in the form of numpy arrays
            seg_mask: segmentation mask uint8 data 
                if paths include 'seg' key else returns None
            affine: data affine (which will be used in saving)
    """


    try:
        brain_mask = nib.load(paths['mask']).get_data()
    except:
        brain_mask = get_brain_mask(paths[list(paths.keys())[0]])

    try:
        seg_mask = np.uint8(nib.load(paths['seg']).get_data())
        seg_mask[(brain_mask != 0)*(seg_mask <= 0)] = 5
        seg_mask[np.where(seg_mask==4)] = 3
        seg_mask[np.where(seg_mask==5)] = 4  ## making an effort to make classes 0,1,2,3,4 rather than 0,1,2,4,5
    except:
        seg_mask = None

    data = {}

    for key in paths.keys():
        nib_vol = nib.load(paths[key])
        affine  = nib_vol.affine 
        vol  = nib_vol.get_data()
        vol  = preprocessing.clip(vol)
        data[key] = preprocessing.standardize(vol,
                                                brain_mask)


    return data, seg_mask, affine


def get_patch(vol, seg = None, coordinate = (0,0,0), size = 64):
    """
        extracts patches from volume and segmentaion
        based on provided coordinate and size

        args
            vol: json with data volumes
            seg: segmentation mask
            coordinate: tuple (<int< of len 3) x, y, x of right top corner
            size: int patch size

        returns
            data patch and segmentation patch if seg is None 
            returns data patch and None 
    """
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
        return data, None


def multilabel_binarize(image_nD, nlabel):
    """
        One hot conversion
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
        enhances the edge information
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
        Data generator object used in pytorch dataloaders

        args
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

        X, y, Emap = self.__data_generation__(index)
        return X, y, Emap

  
    def __data_generation__(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        self.image_size = (self.patch_size, self.patch_size, self.patch_size)
        X = []; y = []; edgeMap = []



        s = len(self.classinfo.keys())
        for bi in range(self.batch_size):
            p = np.random.uniform()
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
            data, mask = self.patch_extractor(vol, seg, 
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
