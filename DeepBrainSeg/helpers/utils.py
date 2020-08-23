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

import SimpleITK as sitk
import numpy as np
import nibabel as nib
import os
from scipy.ndimage.measurements import label
from skimage.morphology import erosion, dilation

#=======================================================================================
# init. brain mask extractor

def imshow(*args,**kwargs):
    """ 
        Handy function to show multiple plots in on row, 
        possibly with different cmaps and titles

        Usage:
        imshow(img1, title="myPlot")
        imshow(img1,img2, title=['title1','title2'])
        imshow(img1,img2, cmap='hot')
        imshow(img1,img2,cmap=['gray','Blues'])
     """

    cmap = kwargs.get('cmap', 'gray')
    title= kwargs.get('title','')
    axis_off = kwargs.get('axis_off','')
    if len(args)==0:
        raise ValueError("No images given to imshow")
    elif len(args)==1:
        plt.title(title)
        plt.imshow(args[0], interpolation='none')
    else:
        n=len(args)
        if type(cmap)==str:
            cmap = [cmap]*n
        if type(title)==str:
            title= [title]*n
        plt.figure(figsize=(n*5,10))
        for i in range(n):
            plt.subplot(1,n,i+1)
            plt.title(title[i])
            plt.imshow(args[i], cmap[i])
            if axis_off: 
              plt.axis('off')  
    plt.show()


def get_brain_mask(t1):
    """
        Provides an alternative for ANTs
        used to extract brain mask given raw image
        Does the task of skull stripping

        t1: T1 MRI volume: height x width x depth

        returns: boolean mask
    """
    from deepbrain import Extractor
    with Extractor() as ext:
        probs = ext.run(t1)
    return probs > 0.5

# Helper Functions.....................

def bbox(vol, pad = 18):
    """
        estimates closest bounding rect for the tumor

        args
            vol: segmentation mask (H x W X D) with tumor
            pad: int; number of zeros to be padded to the 
                    estimated bounding box
        returns
            tuple(<int> of len 6): 
                xmin, xmax, ymin, ymax, zmin, zmax
    """
    vol = np.uint8(vol)
    tumor = np.where(vol>0)

    x_min = np.min(tumor[0])
    y_min = np.min(tumor[1])
    z_min = np.min(tumor[2])

    x_max = np.max(tumor[0])
    y_max = np.max(tumor[1])
    z_max = np.max(tumor[2])

    x_min = max(0, x_min-pad)
    y_min = max(0, y_min-pad)
    z_min = max(0, z_min-pad)
    
    x_max = min(240 - pad, x_max + pad)
    y_max = min(240 - pad, y_max + pad)
    z_max = min(155 - pad, z_max + pad)

    return x_min, x_max, y_min, y_max, z_min, z_max        


def adjust_classes_air_brain_tumour(volume):
    """
        Converts ABL classes to AL class
        as brain is introduced class, 
        it is merged with air in this funciton

        args
            volume: segmentation mask (H x W x D)
        returns
            tensor with same dimension as volume
    """
    volume = np.uint8(volume)
    volume[volume == 1] = 0
    volume[volume == 2] = 1
    return volume


def convert_image(image):
    """
        Rearranging image channels accordingly

        args
            image: W x H x C
        returns
            image: C x W x H
    """
    x= np.float32(image)
    x=np.swapaxes(x,0,2)
    x= np.swapaxes(x,1,2)
    return x


def apply_argmax_to_logits(logits):
    """
        Applies argmax to logits along dimension 0
    
        args
            logits dimensions: nclasses x Height x Width x Depth
        returns
            tensor with dimension Height x Width x Depth (uint8)
    """
    logits = np.argmax(logits, axis=0)         
    return np.uint8(logits)


def adjust_classes(volume):
    """
        readjusts class values in the form of brats 
        data. Removes brain class and reassigns 
        class 3 to class 4

        args
            volume:  segmentaiton mask with Height x Width x Depth
        returns
            tensor with same dimension as input volume
    """
    volume = np.uint8(volume)
    volume[volume == 4] = 0
    volume[volume == 3] = 4
    return volume


def save_volume(volume, affine, path):
    """
        saves given volume into nifty format
        based on given affine

        args
            volume: ndarray to be converted to nifty
            affine: orientation information usually 
                    extracted from raw image
            path: path to save nifty data

    """
    # volume = np.uint8(volume)
    # volume.set_data_dtype(np.uint8)
    volume = nib.Nifti1Image(volume, affine)
    nib.save(volume, path +'.nii.gz')
    pass


def scale_every_slice_between_0_to_255(x):
    """
        renormalizes the intensity values between (0, 256)

        args
            x: ndarray
        returns
            tensor of same dimension as x with p2p (0, 256)
    """ 
    normalized =  255*((x-np.min(x))/(np.max(x)-np.min(x)))
    return normalized
    

def get_whole_tumor(data):
    """
        gets whole tumor region from a given 
        segmentation mask

        args
            data: ndarray (uint8)
        returns
            boolean array with same dimension as data
    """
    return (data > 0)*(data < 4)


def get_tumor_core(data):
    """
        gets tumor core region from a given 
        segmentation mask

        args
            data: ndarray (uint8)
        returns
            boolean array with same dimension as data
    """
    return np.logical_or(data==1,data==4)


def get_enhancing_tumor(data):
    """
        gets enhancing tumor region from a given 
        segmentation mask

        args
            data: ndarray (uint8)
        returns
            boolean array with same dimension as data
    """
    return data == 4


def get_dice_score(prediction, ground_truth):
    """
        Calculates dice coefficient for each class
            
            calculates $$\frac{2*A \cap B}{A \cup B}$$

        args
            prediction: ndarray with predicted segmentation classes
            ground_truth: ndarray with expected segmentation classes

        returns
            tuple(<float> of len 3) whole tumor dice, 
                enhancing tumor dice, and tumor core dice 
    """
    masks=(get_whole_tumor, get_tumor_core, get_enhancing_tumor)
    p    =np.uint8(prediction)
    gt   =np.uint8(ground_truth)
    wt, tc, et=[2*np.sum(func(p)*func(gt)) / (np.sum(func(p)) + np.sum(func(gt))+1e-6) for func in masks]
    return wt, tc, et



def convert5class_logitsto_4class(logits):
    """
        Merges brain class with background

        args
            logits: 5 x H x W x D
        returns
            tensor of dim 4 x H x W x D 
    """
    assert len(logits.shape) == 4
    assert logits.shape[0] == 5

    new_logits = np.zeros((4,)+ logits.shape[1:])
    new_logits[0, :, :, :] = logits[0,:,:,:] + logits[4,:,:,:]
    new_logits[1, :, :, :] = logits[1,:,:,:]
    new_logits[2, :, :, :] = logits[2,:,:,:]
    new_logits[3, :, :, :] = logits[3,:,:,:]
    return new_logits

def combine_logits_GM(x):
    """
        Method to ensemble the predictions of 
        multiple models considers GM over logits
        
            estimates $$(\Prod x_i)^{1/len(x)}$$

        args
            x: list of all predictions
        returns
            tensor of dimension same as x[0]
    """

    assert len(x.shape) == 5
    final = np.ones_like(x[0], dtype='float32')
    for ii in x:
        final = final*ii*10.
    return  (final**(1./x.shape[0]))/10.


def combine_logits_AM(x):
    """
        Method to ensemble the predictions of 
        multiple models considers AM over logits
        
            estimates $$\frac{\Sum x_i}{len(x)}$$

        args
            x: list of all predictions
        returns
            tensor of dimension same as x[0]
    """
    assert len(x[0].shape) == 4
    final = np.zeros_like(x[0])
    for ii in x:
        final = final + ii
    return final * (1./len(x))


def combine_predictions_GM(x):
    """
        Method to ensemble the predictions of 
        multiple models considers GM over predictions
        
            estimates $$(\Prod x_i)^{1/len(x)}$$

        args
            x: list of all predictions
        returns
            tensor of dimension same as x[0]
    """
    x = np.array(x, dtype='float32')
    assert len(x.shape) == 4
    final = np.ones_like(x[0])
    for ii in x:
        final = final*ii
    return np.uint8(final**(1.0/len(x)))


def combine_predictions_AM(x):
    """
        Method to ensemble the predictions of 
        multiple models considers AM over predictions
        
            estimates $$\frac{\Sum x_i}{len(x)}$$

        args
            x: list of all predictions
        returns
            tensor of dimension same as x[0]
    """
    x = np.array(x, dtype='float32')
    assert len(x.shape) == 4
    final = np.zeros_like(x[0])
    for ii in x:
        final = final + ii
    return np.uint8(final * (1.0/len(x)))


def combine_mask_prediction(mask, pred):
    """
        Conbines final inner class predictions 
        with localized predictions

        args
            mask: localized prediction tensor
            pred: inner class prediction tensor
        returns
            tensor of same dimension as mask or pred
    """
    mask[mask == 1]   = 2
    mask[pred == 1]   = 1
    mask[pred == 3]   = 3
    return mask