import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.feature import canny
import skimage.morphology as morph

import torch
from torch.utils.data import Dataset

import sys
sys.path.append('../..')
from helpers.helper import *

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
    affine = nib.load(paths['t1']).affine
    t1ce = nib.load(paths['t1ce']).get_data()

    try:
        brain_mask = nib.load(paths['mask']).get_data()
    except:
        brain_mask = get_ants_mask(ants_path, paths['t1'])

    sege_mask = np.uint8(nib.load(paths['seg']).get_data())
    sege_mask[(brain_mask != 0)*(sege_mask <= 0)] = 5
    sege_mask[np.where(sege_mask==4)] = 3
    sege_mask[np.where(sege_mask==5)] = 4  ## making an effort to make classes 0,1,2,3,4 rather than 0,1,2,4,5

    t1    = normalize(t1, brain_mask)
    t1ce  = normalize(t1ce, brain_mask)
    t2    = normalize(t2, brain_mask)
    flair = normalize(flair, brain_mask)
    data = {}
    data['flair'] = flair
    data['t2'] = t2
    data['t1'] = t1
    data['t1ce'] = t1ce
    return data, sege_mask, affine


def get_patch(vol, seg, coordinate = (0,0,0), size = 64):
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
    seg_mask = seg[coordinate[0]:coordinate[0] + size,
	coordinate[1]:coordinate[1] + size,
	coordinate[2]:coordinate[2] + size]
    return data, seg_mask


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
                 hardmining = False,
        		 transform = None, 
        		 target_transform = None,
                 loader = nii_loader):

        self.csv = pd.read_csv(csv_path)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        """
        patch = self.csv[index].values
        path = patch[0] 
        coordinate = patch[1]
        dice = patch[2]
        data, segmentation = self.loader(path, coordinate)
        data = torch.from_numpy(data).float()
        

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        weight_map = getEdgeEnhancedWeightMap_3D(segmentation)
        return data, segmentation, weight_map, path

    def __len__(self):
        return len(self.csv)


def getDataPaths(path):
    data = pd.read_csv(path)
    imgpaths = data['Paths'].as_matrix()
    np.random.shuffle(imgpaths)
    return imgpaths


if __name__== '__main__':

    csv='./training_patch_info.csv'
    a= getDataPaths(csv)

    dl=ImageFolder(a)
    for i, (x,z,_) in enumerate (dl):
        print ('min',np.min(z),'max',np.max(z), _)
        # print('size',z.long().size())
