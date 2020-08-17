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
import os
import glob
from tqdm import tqdm
from time import gmtime, strftime



class Coregistration(object):
    """
        for data preprocessing converts volume into (1x1x1) resolution
        along with t1ce or mask registration

    """
    def __init__(self):
        self.registration_method = sitk.ImageRegistrationMethod()

        # Similarity metric settings.
        self.registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        self.registration_method.SetMetricSamplingStrategy(self.registration_method.RANDOM)
        self.registration_method.SetMetricSamplingPercentage(0.01)

        self.registration_method.SetInterpolator(sitk.sitkLinear)

        # Optimizer settings.
        self.registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, 
                                                            numberOfIterations=100, 
                                                            convergenceMinimumValue=1e-6, 
                                                            convergenceWindowSize=10)
        self.registration_method.SetOptimizerScalesFromPhysicalShift()

        # Setup for the multi-resolution framework.            
        self.registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
        self.registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
        self.registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    def resize_sitk_3D(self, image_array, outputSize=None, interpolator=sitk.sitkLinear):
        """
            Resample 3D images Image:
            For Labels use nearest neighbour
            For image use
            sitkNearestNeighbor = 1,
            sitkLinear = 2,
            sitkBSpline = 3,
            sitkGaussian = 4,
            sitkLabelGaussian = 5,
        """
        image = image_array
        inputSize = image.GetSize()
        inputSpacing = image.GetSpacing()
        outputSpacing = [1.0, 1.0, 1.0]
        if outputSize:
            outputSpacing[0] = inputSpacing[0] * (inputSize[0] /outputSize[0]);
            outputSpacing[1] = inputSpacing[1] * (inputSize[1] / outputSize[1]);
            outputSpacing[2] = inputSpacing[2] * (inputSize[2] / outputSize[2]);
        else:
            # If No outputSize is specified then resample to 1mm spacing
            outputSize = [0.0, 0.0, 0.0]
            outputSize[0] = int(inputSize[0] * inputSpacing[0] / outputSpacing[0] + .5)
            outputSize[1] = int(inputSize[1] * inputSpacing[1] / outputSpacing[1] + .5)
            outputSize[2] = int(inputSize[2] * inputSpacing[2] / outputSpacing[2] + .5)
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(outputSize)
        resampler.SetOutputSpacing(outputSpacing)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(0)
        image = resampler.Execute(image)
        return image

    def register_patient(self, moving_images, 
                            fixed_image, 
                            save_path,
                            save_transform=True,
                            isotropic=True):
        """
            moving_images : {'key1': path1, 'key2': path2}
            fixed_image :t1c path
            save_path: save path 
        """
        fixed_name = fixed_image.split('/').pop().split('.')[0]
        fixed_image =  sitk.ReadImage(fixed_image, sitk.sitkFloat32)
        coregistration_path = os.path.join(save_path, 'registered')
        isotropic_path = os.path.join(save_path, 'isotropic')
        transform_path = os.path.join(save_path, 'transforms')

        if not os.path.exists(coregistration_path):
            os.makedirs(coregistration_path, exist_ok=True)

        if isotropic:
            if not os.path.exists(isotropic_path):
                os.makedirs(isotropic_path, exist_ok=True)

        if save_transform:
            if not os.path.exists(transform_path):
                os.makedirs(transform_path, exist_ok=True)

        for key in moving_images.keys():
            moving_image = sitk.ReadImage(moving_images[key], sitk.sitkFloat32)
            initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                      moving_image, 
                                                      sitk.VersorRigid3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

            
            self.registration_method.SetInitialTransform(initial_transform, inPlace=False)
            final_transform = self.registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                              sitk.Cast(moving_image, sitk.sitkFloat32))
            
            print("[INFO: DeepBrainSeg] (" + strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + ") " +  'Final metric value: {0}'.format(self.registration_method.GetMetricValue()))
            print("[INFO: DeepBrainSeg] (" + strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + ") " +  'Optimizer\'s stopping condition, {0}'.format(self.registration_method.GetOptimizerStopConditionDescription()))
            
            moving_resampled= sitk.Resample(moving_image, 
                                            fixed_image, 
                                            final_transform, 
                                            sitk.sitkLinear, 0.0, 
                                            moving_image.GetPixelID())
            
            sitk.WriteImage(moving_resampled, os.path.join(coregistration_path, key+'.nii.gz'))
            sitk.WriteTransform(final_transform, os.path.join(transform_path, key+'.tfm'))
            # Write Fixed image in nii.gz
            if isotropic:
                print("[INFO: DeepBrainSeg] (" + strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + ") " +  'converting to isotropic volume')
                moving_resized = self.resize_sitk_3D(moving_resampled)
                sitk.WriteImage(moving_resized, os.path.join(isotropic_path, key+'.nii.gz'))

        sitk.WriteImage(fixed_image, os.path.join(coregistration_path, fixed_name+'.nii.gz'))            
        if isotropic:
            fixed_resized = self.resize_sitk_3D(fixed_image)
            sitk.WriteImage(fixed_resized, os.path.join(isotropic_path, fixed_name+'.nii.gz'))
