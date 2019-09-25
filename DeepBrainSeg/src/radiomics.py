import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import six
import radiomics
from tqdm import tqdm
from radiomics import firstorder, glcm, imageoperations, glrlm, glszm, ngtdm, gldm, getTestCase

class ExtractRadiomicFeatures():
    def __init__(input_image, 
                    input_mask=None, 
                    save_path=None, 
                    seq='Flair',
                    class_ = 'ET',
                    all_=True):
        
        self.input_image = input_image
        if not input_mask:
            self.input_mask = np.ones(tuple(list(self.input_image.shape)[:-1]))
        else: self.input_mask = input_mask

        self.save_path = save_path
        self.seq = seq
        self.all_ = all_
        self.class_ = class_
        self.feat_dict = {}


    def first_order(self):

        feat_dict = {}
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(img,GT)
        firstOrderFeatures.enableAllFeatures()
        firstOrderFeatures.execute()          
        for (key,val) in six.iteritems(firstOrderFeatures.featureValues):
            if self.all_: 
                self.feat_dict[self.seq + "_" + self.class_ + '_' + key] = val
            else: 
                feat_dict[self.seq + "_" + self.class_ + "_" + key] = val

        df = pd.DataFrame(feat_dict)
        if self.save_path:
            df.to_csv(os.path.join(self.save_path, 'firstorder_features.csv'), index=False)

        return df


    def glcm_features(self):

        glcm_dict = {}
        GLMFeatures = glcm.RadiomicsGLCM(img,GT)
        GLCMFeatures.enableAllFeatures()
        GLCMFeatures.execute()
        for (key,val) in six.iteritems(GLCMFeatures.featureValues):
            if self.all_: 
                self.feat_dict[self.seq + "_" + self.class_ + '_' + key] = val
            else: 
                glcm_dict[self.seq + "_" + self.class_ + "_" + key] = val

        df = pd.DataFrame(glcm_dict)
        if self.save_path:
            df.to_csv(os.path.join(self.save_path, 'glcm_features.csv'), index=False)

        return df


    def glszm_features(self):
        
        glszm_dict = {}
        GLSZMFeatures = glszm.RadiomicsGLSZM(img,GT)
        GLSZMFeatures.enableAllFeatures()  # On the feature class level, all features are disabled by default.
        GLSZMFeatures.execute()
        for (key,val) in six.iteritems(GLSZMFeatures.featureValues):
            if self.all_: 
                self.feat_dict[self.seq + "_" + self.class_ + '_' + key] = val
            else: 
                glszm_dict[self.seq + "_" + self.class_ + "_" + key] = val

        df = pd.DataFrame(glszm_dict)
        if self.save_path:
            df.to_csv(os.path.join(self.save_path, 'glszm_features.csv'), index=False)


        return df
    
    
    def glrlm_features(self):


        glrlm_dict = {}
        GLRLMFeatures = glrlm.RadiomicsGLRLM(img,GT)
        GLRLMFeatures.enableAllFeatures()  # On the feature class level, all features are disabled by default.
        GLRLMFeatures.execute()
        for (key,val) in six.iteritems(GLRLMFeatures.featureValues):
            if self.all_: 
                self.feat_dict[self.seq + "_" + self.class_ + '_' + key] = val
            else: 
                glrlm_dict[self.seq + "_" + self.class_ + "_" + key] = val

        df = pd.DataFrame(glrlm_dict)
        if self.save_path:
            df.to_csv(os.path.join(self.save_path, 'glrlm_features.csv'), index=False)

    
        return df
    
    
    def ngtdm_features(self):
        
        ngtdm_dict {}
        NGTDMFeatures = ngtdm.RadiomicsNGTDM(img,GT)
        NGTDMFeatures.enableAllFeatures()  # On the feature class level, all features are disabled by default.
        NGTDMFeatures.execute()
        for (key,val) in six.iteritems(NGTDMFeatures.featureValues):
            if self.all_: 
                self.feat_dict[self.seq + "_" + self.class_ + '_' + key] = val
            else: 
                ngtdm_dict[self.seq + "_" + self.class_ + "_" + key] = val

        df = pd.DataFrame(ngtdm_dict)
        if self.save_path:
            df.to_csv(os.path.join(self.save_path, 'ngtdm_features.csv'), index=False)
    
        return df

    def gldm_features(self):

        gldm_dict = {}
        GLDMFeatures = gldm.RadiomicsGLDM(img,GT)
        GLDMFeatures.enableAllFeatures()  # On the feature class level, all features are disabled by default.
        GLDMFeatures.execute()
        for (key,val) in six.iteritems(GLDMFeatures.featureValues):

            if self.all_: 
                self.feat_dict[self.seq + "_" + self.class_ + '_' + key] = val
            else: 
                gldm_dict[self.seq + "_" + self.class_ + "_" + key] = val

        df = pd.DataFrame(gldm_dict)
        if self.save_path:
            df.to_csv(os.path.join(self.save_path, 'gldm_features.csv'), index=False)

        return df

    
    def all_features(self):

        _ = self.first_order()
        _ = glcm_features()
        _ = glszm_features()
        _ = glrm_features()
        _ = gldm_features()
        _ = ngtdm_features()
        
        df = pd.DataFrame(self.feat_dict)

        if self.save_path:
            df.to_csv(os.path.join(self.save_path, 'all_features.csv'), index=False)

        return df
