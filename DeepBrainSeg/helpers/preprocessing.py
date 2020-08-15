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
import SimpleITK as sitk

def clip(x, q=90):
    """
        used to eliminate intensity outliers

        q: percentile for max_value
    """
    x = np.nan_to_num(x)
    
    min_val = 0
    max_val = np.percentile(
        x, q,
        axis=None
    )
    
    x = np.clip(
        x,
        a_min=min_val,
        a_max=max_val
    )
    return x


def standardize(img, 
                mask = None, 
                median = False):
    """
        standardizes the volume by offsetting the intensities
        w.r.t mean or median on brain intensities
        and normalize with the standard deviation of 
        intensity values

        img: ndarray, image used for analysis
        mask: ndarray, brain mask
        median: boolean, arguments allowes us to use 
                median for offsetting intensity values
    """

    if median: offset = np.percentile(img[mask != 0], 
                                        q=50, 
                                        axis=None)
    else: offset = np.mean(img[mask != 0])


    std  = np.std(img[mask != 0])
    return (img - offset)/std


def normalize(img, 
                mask =None):
    """
        squeezes the intensity values between 0-1   
            (x - min(x)) / (max(x) - min(x))
        
        img: ndarray, image used for analysis
        mask: ndarray, brain mask
    """

    min_vol = np.min(img[mask != 0])
    max_vol = np.max(img[mask != 0])
    
    return (img - min_vol) / (max_vol - min_vol + 1e-3)


def resample3D(img, 
                outputSize=None, 
                interpolator=sitk.sitkBSpline):
    """
        Resample 3D images Image, interpolates or 
        subsamples the volume based on the arguments 
        provided


        img: ndarray, 3D volume
        outputSize: tuple (x,y,z), required dimension 
            of the image in output
        interpolator: int (0, ..., 5), type of interpolator 
            For Labels use nearest neighbour interpolation
            For image can use any: 
                sitkNearestNeighbor = 1,
                sitkLinear = 2,
                sitkBSpline = 3,
                sitkGaussian = 4,
                sitkLabelGaussian = 5, 
    """
    volume = sitk.GetImageFromArray(img)
    inputSize = volume.GetSize()
    inputSpacing = volume.GetSpacing()
    outputSpacing = [1.0, 1.0, 1.0]

    if outputSize:
        # based on provided information and aspect ratio of the
        # original volume
        outputSpacing[0] = inputSpacing[0] * (inputSize[0] / outputSize[0])
        outputSpacing[1] = inputSpacing[1] * (inputSize[1] / outputSize[1])
        outputSpacing[2] = inputSpacing[2] * (inputSize[2] / outputSize[2])
    else:
        # If No outputSize is specified then resample to 1mm spacing
        outputSize = [0.0, 0.0, 0.0]
        outputSize[0] = int(
            inputSize[0] * inputSpacing[0] / outputSpacing[0] + .5)
        outputSize[1] = int(
            inputSize[1] * inputSpacing[1] / outputSpacing[1] + .5)
        outputSize[2] = int(
            inputSize[2] * inputSpacing[2] / outputSpacing[2] + .5)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(outputSize)
    resampler.SetOutputSpacing(outputSpacing)
    resampler.SetOutputOrigin(volume.GetOrigin())
    resampler.SetOutputDirection(volume.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    volume = resampler.Execute(volume)
    resampled_volume = sitk.GetArrayFromImage(volume)
    return resampled_volume
