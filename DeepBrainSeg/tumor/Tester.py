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


import torch
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from torch.autograd import Variable
from skimage.transform import resize
from torchvision import transforms
from time import gmtime, strftime

from tqdm import tqdm
import pdb
import os

from . import maybe_download
from ..helpers import utils
from ..helpers import postprocessing
from ..helpers import preprocessing
from .. import brainmask

from os.path import expanduser
home = expanduser("~")

#========================================================================================
class tumorSeg():
    """
        class performs segmentation for a given sequence of patient data.
        to main platform for segmentation mask estimation
            one for the patient data in brats format
            other with any random format
        step followed for in estimation of segmentation mask
            1. ABLnet for reducing false positives outside the brain
                Air Brain Lesson model (2D model, 103 layered)
            2. BNet3Dnet 3D network for inner class classification
                Dual Path way network
            3. Tir3Dnet 57 layered 3D convolutional network for inner class 
                classification
        more on training details and network information:
        (https://link.springer.com/chapter/10.1007/978-3-030-11726-9_43<Paste>)
	
	=========================
	
	quick: True (just evaluates on Dual path network (BNet3D)
		else copmutes an ensumble over all four networks	
    """
    def __init__(self, 
                    quick = False,
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

        map_location = device
        #========================================================================================

        ckpt_tir3D    = os.path.join(home, '.DeepBrainSeg/BestModels/tumor_Tramisu_FC57_3D.pth.tar')
        ckpt_BNET3D   = os.path.join(home, '.DeepBrainSeg/BestModels/tumor_BrainNet_3D.pth.tar')
        ckpt_ABL      = os.path.join(home, '.DeepBrainSeg/BestModels/tumor_ABL_2D.pth.tar')

        #========================================================================================
        # air brain lesion segmentation..............
        from .models.modelABL import FCDenseNet103

        self.ABLnclasses = 3
        self.ABLnet = FCDenseNet103(n_classes = self.ABLnclasses) ## intialize the graph
        maybe_download(ckpt_ABL)
        saved_parms=torch.load(ckpt_ABL, map_location=map_location) 
        self.ABLnet.load_state_dict(saved_parms['state_dict']) ## fill the model with trained params
        print ("================================ ABLNET2D Loaded ==============================")
        self.ABLnet.eval()
        self.ABLnet = self.ABLnet.to(device)

        #========================================================================================
        # Tir3D model...................
        from .models.modelTir3D import FCDenseNet57

        self.T3Dnclasses = 5
        self.Tir3Dnet = FCDenseNet57(self.T3Dnclasses)
        maybe_download(ckpt_tir3D)
        ckpt = torch.load(ckpt_tir3D, map_location=map_location)
        self.Tir3Dnet.load_state_dict(ckpt['state_dict'])
        print ("=============================== TIRNET2D Loaded ==============================")
        self.Tir3Dnet.eval()
        self.Tir3Dnet = self.Tir3Dnet.to(device)

        if not quick:
            
            # BrainNet3D model......................
            from .models.model3DBNET import BrainNet_3D_Inception
            self.B3Dnclasses = 5
            self.BNET3Dnet = BrainNet_3D_Inception()
            maybe_download(ckpt_BNET3D)
            ckpt = torch.load(ckpt_BNET3D, map_location=map_location)
            self.BNET3Dnet.load_state_dict(ckpt['state_dict'])
            print ("================================ KAMNET3D Loaded ==============================")
            self.BNET3Dnet.eval()
            self.BNET3Dnet = self.BNET3Dnet.to(device)


        #========================================================================================

        self.device = device
        self.quick = quick


    def get_localization(self, t1, t1ce, t2, flair, brain_mask):
        """
		ABLnetwork output, finds the brain, Whole tumor region

		t1_v       = t1 volume (numpy array)
		t1c_v      = t1c volume (numpy array)
		t2_v       = t2 volume (numpy array)
		flair_v    = flair volume (numpy array)
		brain_mask = brain, whole tumor mask (numpy array, output of ANTs pieline)
        """

        t1    = preprocessing.standardize(t1,    brain_mask)
        t1ce  = preprocessing.standardize(t1ce,   brain_mask)
        t2    = preprocessing.standardize(t2,    brain_mask)
        flair = preprocessing.standardize(flair, brain_mask)

        generated_output_logits = np.empty((self.ABLnclasses, flair.shape[0], flair.shape[1], flair.shape[2]))

        for _slice_ in tqdm(range(flair.shape[2])):
            flair_slice = np.transpose(flair[:,:,_slice_])
            t2_slice    = np.transpose(t2[:,:,_slice_])
            t1ce_slice  = np.transpose(t1ce[:,:,_slice_])
            t1_slice    = np.transpose(t1[:,:,_slice_])  
                          
            array        = np.zeros((flair_slice.shape[0],flair_slice.shape[1],4))
            array[:,:,0] = flair_slice
            array[:,:,1] = t2_slice
            array[:,:,2] = t1ce_slice
            array[:,:,3] = t1_slice
                

            transformed_array = torch.from_numpy(utils.convert_image(array)).float()
            transformed_array = transformed_array.unsqueeze(0) ## neccessary if batch size == 1
            transformed_array = transformed_array.to(self.device)
            logits            = self.ABLnet(transformed_array).detach().cpu().numpy()# 3 x 240 x 240  
            generated_output_logits[:,:,:, _slice_] = logits.transpose(0, 1, 3, 2)

        final_pred  = utils.apply_argmax_to_logits(generated_output_logits)
        final_pred  = postprocessing.class_wise_cc(final_pred)
        final_pred  = utils.adjust_classes_air_brain_tumour(np.uint8(final_pred))

        return np.uint8(final_pred)


    def inner_class_classification_with_logits_NCube(self, t1, 
                                                        t1ce, t2, flair, 
                                                        brain_mask, mask, N = 64):
        """
		output of 3D tiramisu model (tir3Dnet)

		mask = numpy array output of ABLnet 
		N = patch size during inference
        """

        t1    = preprocessing.standardize(t1,    brain_mask)
        t1ce  = preprocessing.standardize(t1ce,   brain_mask)
        t2    = preprocessing.standardize(t2,    brain_mask)
        flair = preprocessing.standardize(flair, brain_mask)

        vol = {}
        vol['t1'] = t1
        vol['t2'] = t2
        vol['t1ce'] = t1ce 
        vol['flair'] = flair

        s = N//4
        for key in vol.keys():
            vol[key] = np.pad(vol[key], ((s, s), (s, s), (s,s))) 
          
        shape = vol['t1'].shape # to exclude batch_size
        final_prediction = np.zeros((self.T3Dnclasses, shape[0], shape[1], shape[2]))

        x_min, x_max, y_min, y_max, z_min, z_max = 0, shape[0], 0, shape[1], 0, shape[2]
        x_min, x_max, y_min, y_max, z_min, z_max = x_min, min(shape[0] - N, x_max), y_min, min(shape[1] - N, y_max), z_min, min(shape[2] - N, z_max)
        with torch.no_grad():
            for x in tqdm(range(x_min, x_max, N//2)):
                for y in range(y_min, y_max, N//2):
                    for z in range(z_min, z_max, N//2):
                        high = np.zeros((1, 4, N, N, N))

                        high[0, 0, :, :, :] = vol['flair'][x:x+N, y:y+N, z:z+N]
                        high[0, 1, :, :, :] = vol['t2'][x:x+N, y:y+N, z:z+N]
                        high[0, 2, :, :, :] = vol['t1'][x:x+N, y:y+N, z:z+N]
                        high[0, 3, :, :, :] = vol['t1ce'][x:x+N, y:y+N, z:z+N]

                        high = Variable(torch.from_numpy(high)).to(self.device).float()
                        pred = torch.nn.functional.softmax(self.Tir3Dnet(high).detach().cpu())
                        pred = pred.data.numpy()

                        final_prediction[:, x+s:x+3*s, y+s:y+3*s, z+s:z+3*s] = pred[0][:, s:-s, s:-s, s:-s]

        final_prediction = final_prediction[:, s:-s, s:-s, s:-s]

        return final_prediction


    def inner_class_classification_with_logits_DualPath(self, t1, 
                                                            t1ce, t2, flair, 
                                                            brain_mask, mask=None, 
                                                            prediction_size = 9):

        """
		output of BNet3D 

		prediction_size = mid inference patch size 
        """

        t1    = preprocessing.standardize(t1,    brain_mask)
        t1ce  = preprocessing.standardize(t1ce,   brain_mask)
        t2    = preprocessing.standardize(t2,    brain_mask)
        flair = preprocessing.standardize(flair, brain_mask)

        shape = t1.shape # to exclude batch_size
        final_prediction = np.zeros((self.B3Dnclasses, shape[0], shape[1], shape[2]))

        x_min, x_max, y_min, y_max, z_min, z_max = utils.bbox(mask, pad = 2*prediction_size)

        # obtained by aspect ratio calculation
        high_res_size   = prediction_size + 16
        resize_to       = int(prediction_size ** 0.5) + 16   
        low_res_size    = int(51*resize_to/19) 

        hl_pad = (high_res_size - prediction_size)//2
        hr_pad = hl_pad + prediction_size

        ll_pad = (low_res_size - prediction_size)//2
        lr_pad = ll_pad + prediction_size

        for x in tqdm(range(x_min, x_max - prediction_size, prediction_size)):
            for y in (range(y_min, y_max - prediction_size, prediction_size)):
                for z in (range(z_min, z_max - prediction_size, prediction_size)):
                    high = np.zeros((1, 4, high_res_size, high_res_size, high_res_size))
                    low  = np.zeros((1, 4, low_res_size, low_res_size, low_res_size))
                    low1  = np.zeros((1, 4, resize_to, resize_to, resize_to))

                    high[0, 0], high[0, 1], high[0, 2], high[0, 3] = high[0, 0] + flair[0,0,0], high[0, 1] + t2[0,0,0], high[0, 2] + t1[0,0,0], high[0, 2] + t1ce[0,0,0]
                    low[0, 0], low[0, 1], low[0, 2], low[0, 3]     = low[0, 0] + flair[0,0,0], low[0, 1] + t2[0,0,0], low[0, 2] + t1[0,0,0], low[0, 2] + t1ce[0,0,0]
                    low1[0, 0], low1[0, 1], low1[0, 2], low1[0, 3] = low1[0, 0] + flair[0,0,0], low1[0, 1] + t2[0,0,0], low1[0, 2] + t1[0,0,0], low1[0, 2] + t1ce[0,0,0]


                    # =========================================================================
                    vxf, vxt = max(0, x-hl_pad), min(shape[0], x+hr_pad)
                    vyf, vyt = max(0, y-hl_pad), min(shape[1], y+hr_pad)
                    vzf, vzt = max(0, z-hl_pad), min(shape[2], z+hr_pad)

                    txf, txt = max(0, hl_pad-x), max(0, hl_pad-x) + vxt - vxf
                    tyf, tyt = max(0, hl_pad-y), max(0, hl_pad-y) + vyt - vyf
                    tzf, tzt = max(0, hl_pad-z), max(0, hl_pad-z) + vzt - vzf

                    high[0, 0, txf:txt, tyf:tyt, tzf:tzt] = flair[vxf:vxt, vyf:vyt, vzf:vzt]
                    high[0, 1, txf:txt, tyf:tyt, tzf:tzt] = t2[vxf:vxt, vyf:vyt, vzf:vzt]
                    high[0, 2, txf:txt, tyf:tyt, tzf:tzt] = t1[vxf:vxt, vyf:vyt, vzf:vzt]
                    high[0, 3, txf:txt, tyf:tyt, tzf:tzt] = t1ce[vxf:vxt, vyf:vyt, vzf:vzt]

                    # =========================================================================
                    vxf, vxt = max(0, x-ll_pad), min(shape[0], x+lr_pad)
                    vyf, vyt = max(0, y-ll_pad), min(shape[1], y+lr_pad)
                    vzf, vzt = max(0, z-ll_pad), min(shape[2], z+lr_pad)

                    txf, txt = max(0, ll_pad-x), max(0, ll_pad-x) + vxt - vxf
                    tyf, tyt = max(0, ll_pad-y), max(0, ll_pad-y) + vyt - vyf
                    tzf, tzt = max(0, ll_pad-z), max(0, ll_pad-z) + vzt - vzf

                    low[0, 0, txf:txt, tyf:tyt, tzf:tzt]  = flair[vxf:vxt, vyf:vyt, vzf:vzt]
                    low[0, 1, txf:txt, tyf:tyt, tzf:tzt]  = t2[vxf:vxt, vyf:vyt, vzf:vzt]
                    low[0, 2, txf:txt, tyf:tyt, tzf:tzt]  = t1[vxf:vxt, vyf:vyt, vzf:vzt]
                    low[0, 3, txf:txt, tyf:tyt, tzf:tzt]  = t1ce[vxf:vxt, vyf:vyt, vzf:vzt]

                    # =========================================================================     
                    low1[0] = [resize(low[0, i, :, :, :], (resize_to, resize_to, resize_to)) for i in range(4)]

                    high = Variable(torch.from_numpy(high)).to(self.device).float()
                    low1  = Variable(torch.from_numpy(low1)).to(self.device).float()
                    pred  = torch.nn.functional.softmax(self.BNET3Dnet(high, low1, 
                                                pred_size=prediction_size).detach().cpu())
                    pred  = pred.numpy()

                    final_prediction[:, x:x + prediction_size, 
                                        y:y+prediction_size, 
                                        z:z+prediction_size] = pred[0]
        return final_prediction


    def get_segmentation(self, 
                        t1_path, 
                        t2_path, 
                        t1ce_path, 
                        flair_path, 
                        save_path = None):
        """
		Generates segmentation for the data not in brats format
		
		if save_path provided function saves the prediction with 
			DeepBrainSeg_Prediction.nii.qz name in the provided 
			directory 
		returns: segmentation mask
        """
        t1 = nib.load(t1_path).get_data()
        t2 = nib.load(t2_path).get_data()
        t1ce = nib.load(t1ce_path).get_data()
        flair = nib.load(flair_path).get_data()
        affine = nib.load(flair_path).affine

        brain_mask = brainmask.get_brain_mask(t2_path)
        mask  =  self.get_localization(t1, t1ce, t2, flair, brain_mask)

        if self.quick:        
            final_predictionTir3D_logits  = self.inner_class_classification_with_logits_NCube(t1, t1ce, t2, flair, brain_mask, mask)
            final_prediction_array        = np.array([final_predictionTir3D_logits])
        else:
            final_predictionTir3D_logits  = self.inner_class_classification_with_logits_NCube(t1, t1ce, t2, flair, brain_mask, mask)
            final_predictionBNET3D_logits = self.inner_class_classification_with_logits_DualPath(t1, t1ce, t2, flair, brain_mask, mask)
            final_prediction_array        = np.array([final_predictionTir3D_logits, final_predictionBNET3D_logits])


        final_prediction_logits = utils.combine_logits_AM(final_prediction_array)
        final_prediction_logits = utils.convert5class_logitsto_4class(final_prediction_logits)
        # final_pred              = utils.apply_argmax_to_logits(final_prediction_logits)
        final_pred              = postprocessing.densecrf(final_prediction_logits)
        final_pred              = postprocessing.class_wise_cc(final_pred)
        final_pred              = utils.combine_mask_prediction(mask, final_pred)
        final_pred              = utils.adjust_classes(final_pred)


        if save_path:
            os.makedirs(save_path, exist_ok=True)
            utils.save_volume(final_pred, affine, os.path.join(save_path, 'DeepBrainSeg_Prediction'))

        return final_pred


    def get_segmentation_brats(self, 
                                path,
                                save = True):
        """
		Generates segmentation for the data in BraTs format

		if save True saves the prediction in the save directory 
			in the patients data path
		
		returns : segmentation mask
        """

        name  = path.split("/")[-1] + "_"
        flair_path =  os.path.join(path, name + 'flair.nii.gz')
        t1_path    =  os.path.join(path, name + 't1.nii.gz')
        t1ce_path  =  os.path.join(path, name + 't1ce.nii.gz')
        t2_path    =  os.path.join(path, name + 't2.nii.gz')
        
        print ("[INFO: DeepBrainSeg] (" + strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + ") Working on: ", path)

        return self.get_segmentation(t1_path, t2_path, t1ce_path, flair_path, path)



# ========================================================================================

if __name__ == '__main__':
    ext = deepSeg(True)
    ext.get_segmentation_brats('../../sample_volume/Brats18_CBICA_AVG_1/')
