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
import SimpleITK as sitk
import six
from tqdm import tqdm
import pickle
from radiomics import firstorder, glcm, imageoperations, glrlm, glszm, ngtdm, gldm, getTestCase

class ExtractRadiomicFeatures():
    def __init__(self, input_image, 
                    input_mask=None, 
                    save_path=None, 
                    seq='Flair',
                    class_ = 'ET',
                    all_=True):
        
        self.input_image = input_image
        if input_mask is None:
            self.input_mask = np.ones(tuple(list(self.input_image.shape)[:-1]))
        else: self.input_mask = input_mask
        
        self.img = sitk.GetImageFromArray(self.input_image)
        self.GT  = sitk.GetImageFromArray(self.input_mask)
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.seq = seq
        self.all_ = all_
        self.class_ = class_
        self.feat_dict = {}


    def write(self, data, path):
        with open(path, "wb") as write_file:
            pickle.dump(data, write_file)


    def first_order(self):

        feat_dict = {}
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(self.img, self.GT)
        firstOrderFeatures.enableAllFeatures()
        firstOrderFeatures.execute()          
        for (key,val) in six.iteritems(firstOrderFeatures.featureValues):
            if self.all_: 
                self.feat_dict[self.seq + "_" + self.class_ + '_' + key] = val
            else: 
                feat_dict[self.seq + "_" + self.class_ + "_" + key] = val

        if self.save_path and (not self.all_):
            self.write(feat_dict, os.path.join(self.save_path, 'firstorder_features.pickle'))

        return feat_dict


    def glcm_features(self):

        glcm_dict = {}
        GLCMFeatures = glcm.RadiomicsGLCM(self.img, self.GT)
        GLCMFeatures.enableAllFeatures()
        GLCMFeatures.execute()
        for (key,val) in six.iteritems(GLCMFeatures.featureValues):
            if self.all_: 
                self.feat_dict[self.seq + "_" + self.class_ + '_' + key] = val
            else: 
                glcm_dict[self.seq + "_" + self.class_ + "_" + key] = val

        if self.save_path and (not self.all_):
            self.write(glcm_dict, os.path.join(self.save_path, 'glcm_features.pickle'))

        return glcm_dict


    def glszm_features(self):
        
        glszm_dict = {}
        GLSZMFeatures = glszm.RadiomicsGLSZM(self.img, self.GT)
        GLSZMFeatures.enableAllFeatures()  # On the feature class level, all features are disabled by default.
        GLSZMFeatures.execute()
        for (key,val) in six.iteritems(GLSZMFeatures.featureValues):
            if self.all_: 
                self.feat_dict[self.seq + "_" + self.class_ + '_' + key] = val
            else: 
                glszm_dict[self.seq + "_" + self.class_ + "_" + key] = val

        if self.save_path and (not self.all_):
            self.write(glszm_dict, os.path.join(self.save_path, 'glszm_features.pickle'))


        return glszm_dict
    
    
    def glrlm_features(self):


        glrlm_dict = {}
        GLRLMFeatures = glrlm.RadiomicsGLRLM(self.img, self.GT)
        GLRLMFeatures.enableAllFeatures()  # On the feature class level, all features are disabled by default.
        GLRLMFeatures.execute()
        for (key,val) in six.iteritems(GLRLMFeatures.featureValues):
            if self.all_: 
                self.feat_dict[self.seq + "_" + self.class_ + '_' + key] = val
            else: 
                glrlm_dict[self.seq + "_" + self.class_ + "_" + key] = val

        if self.save_path and (not self.all_):
            self.write(glrlm_dict, os.path.join(self.save_path, 'glrlm_features.pickle'))

    
        return glrlm_dict
    
    
    def ngtdm_features(self):
        
        ngtdm_dict = {}
        NGTDMFeatures = ngtdm.RadiomicsNGTDM(self.img, self.GT)
        NGTDMFeatures.enableAllFeatures()  # On the feature class level, all features are disabled by default.
        NGTDMFeatures.execute()
        for (key,val) in six.iteritems(NGTDMFeatures.featureValues):
            if self.all_: 
                self.feat_dict[self.seq + "_" + self.class_ + '_' + key] = val
            else: 
                ngtdm_dict[self.seq + "_" + self.class_ + "_" + key] = val

        if self.save_path and (not self.all_):
            self.write(ngtdm_dict, os.path.join(self.save_path, 'ngtdm_features.pickle'))
    
        return ngtdm_dict

    def gldm_features(self):

        gldm_dict = {}
        GLDMFeatures = gldm.RadiomicsGLDM(self.img, self.GT)
        GLDMFeatures.enableAllFeatures()  # On the feature class level, all features are disabled by default.
        GLDMFeatures.execute()
        for (key,val) in six.iteritems(GLDMFeatures.featureValues):

            if self.all_: 
                self.feat_dict[self.seq + "_" + self.class_ + '_' + key] = val
            else: 
                gldm_dict[self.seq + "_" + self.class_ + "_" + key] = val

        if self.save_path and (not self.all_):
            self.write(gldm_dict, os.path.join(self.save_path, 'gldm_features.pickle'))

        return gldm_dict

    
    def all_features(self):

        _ = self.first_order()
        _ = self.glcm_features()
        _ = self.glszm_features()
        _ = self.glrlm_features()
        _ = self.gldm_features()
        _ = self.ngtdm_features()
        
        if self.save_path:
            self.write(self.feat_dict, os.path.join(self.save_path, 'all_features.pickle'))

        return self.feat_dict
