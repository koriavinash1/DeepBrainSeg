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

import numpy as np
from scipy.ndimage import uniform_filter, maximum_filter
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (unary_from_softmax, 
                                create_pairwise_bilateral, 
                                create_pairwise_gaussian)
from scipy.ndimage.measurements import label
from skimage.morphology import erosion, dilation


def densecrf(logits):
    """
    applies coditional random fields on predictions
    The idea is consider the nbr voxels in making 
    class prediction of current pixel
    
    refer CRF and MRF papers for more theoretical idea

    logits: Nb_classes x Height x Width x Depth
    """
    shape = logits.shape[1:]
    new_image = np.empty(shape)
    d = dcrf.DenseCRF(np.prod(shape), logits.shape[0])
    U = unary_from_softmax(logits)
    d.setUnaryEnergy(U)
    feats = create_pairwise_gaussian(sdims=(1.0, 1.0, 1.0), shape=shape)
    d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5) 
    new_image = np.argmax(Q, axis=0).reshape((shape[0], shape[1],shape[2]))
    return new_image

def connected_components(voxels, threshold=12000):
    """
    This clusters entire segmentations into multiple clusters
    and considers significant cluster for further analysis

    voxels: np.uint8 height x width x depth
    threshold: number of pixels in cluster to 
        consider it as significant
    """

    c,n = label(voxels)
    nums = np.array([np.sum(c==i) for i in range(1, n+1)])
    selected_components = nums>threshold
    selected_components[np.argmax(nums)] = True
    mask = np.zeros_like(voxels)

    for i,select in enumerate(selected_components):
        if select:
            mask[c==(i+1)]=1
    return mask*voxels


def class_wise_cc(logits):
    """
    Applies connected components on class wise slices
    
    logits dimension: nclasses, width, height, depth
    """
    return_ = np.zeros_like(logits)
    for class_ in range(logits.shape[0]):
        return_[class_, :, :, :] = connected_components(logits[class_, :, :, :])

    return return_
