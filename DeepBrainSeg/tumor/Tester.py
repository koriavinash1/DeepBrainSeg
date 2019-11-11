#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# author: Avinash Kori
# contact: koriavinash1@gmail.com

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

from ..helpers.helper import *
from os.path import expanduser
home = expanduser("~")

#========================================================================================
# prediction functions.....................
bin_path = os.path.join('/opt/ANTs/bin/')
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
            3. MNet2D 57 layered convolutional network for inner class
                classification
            4. Tir3Dnet 57 layered 3D convolutional network for inner class 
                classification
        more on training details and network information:
        (https://link.springer.com/chapter/10.1007/978-3-030-11726-9_43<Paste>)
	
	=========================
	
	quick: True (just evaluates on Dual path network (BNet3D)
		else copmutes an ensumble over all four networks	
    """
    def __init__(self, 
                    quick = False,
                    ants_path = bin_path):


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = "cpu"

        map_location = device
        #========================================================================================

        ckpt_tir2D    = os.path.join(home, '.DeepBrainSeg/BestModels/Tramisu_2D_FC57_best_loss.pth.tar')
        ckpt_tir3D    = os.path.join(home, '.DeepBrainSeg/BestModels/Tramisu_3D_FC57_best_acc.pth.tar')
        ckpt_BNET3D   = os.path.join(home, '.DeepBrainSeg/BestModels/BrainNet_3D_best_acc.pth.tar')
        ckpt_ABL      = os.path.join(home, '.DeepBrainSeg/BestModels/ABL_CE_best_model_loss_based.pth.tar')

        #========================================================================================
        # air brain lesion segmentation..............
        from .models.modelABL import FCDenseNet103

        self.ABLnclasses = 3
        self.ABLnet = FCDenseNet103(n_classes = self.ABLnclasses) ## intialize the graph
        saved_parms=torch.load(ckpt_ABL, map_location=map_location) 
        self.ABLnet.load_state_dict(saved_parms['state_dict']) ## fill the model with trained params
        print ("=================================== ABLNET2D Loaded =================================")
        self.ABLnet.eval()
        self.ABLnet = self.ABLnet.to(device)

        #========================================================================================

        # Tir2D net.......................
        from .models.modelTir2D import FCDenseNet57
        self.Mnclasses = 4
        self.MNET2D = FCDenseNet57(self.Mnclasses)
        ckpt = torch.load(ckpt_tir2D, map_location=map_location)
        self.MNET2D.load_state_dict(ckpt['state_dict'])
        print ("=================================== MNET2D Loaded ===================================")
        self.MNET2D.eval()
        self.MNET2D = self.MNET2D.to(device)

        #========================================================================================

        if not quick:
            
            # BrainNet3D model......................
            from .models.model3DBNET import BrainNet_3D_Inception
            self.B3Dnclasses = 5
            self.BNET3Dnet = BrainNet_3D_Inception()
            ckpt = torch.load(ckpt_BNET3D, map_location=map_location)
            self.BNET3Dnet.load_state_dict(ckpt['state_dict'])
            print ("=================================== KAMNET3D Loaded =================================")
            self.BNET3Dnet.eval()
            self.BNET3Dnet = self.BNET3Dnet.to(device)

            #========================================================================================
            # Tir3D model...................
            from .models.modelTir3D import FCDenseNet57

            self.T3Dnclasses = 5
            self.Tir3Dnet = FCDenseNet57(self.T3Dnclasses)
            ckpt = torch.load(ckpt_tir3D, map_location=map_location)
            self.Tir3Dnet.load_state_dict(ckpt['state_dict'])
            print ("================================== TIRNET2D Loaded =================================")
            self.Tir3Dnet.eval()
            self.Tir3Dnet = self.Tir3Dnet.to(device)


        #========================================================================================

        self.device = device
        self.quick = quick
        self.ants_path = ants_path


    def get_ants_mask(self, t1_path):
        """
		We make use of ants framework for generalized skull stripping
		
		t1_path: t1 volume path (str)
		saves the mask in the same location as t1 data directory
		returns: maskvolume (numpy uint8 type) 
        """
        mask_path = os.path.join(os.path.dirname(t1_path), 'mask.nii.gz')
        os.system(self.ants_path +'ImageMath 3 '+ mask_path +' Normalize '+ t1_path)
        os.system(self.ants_path +'ThresholdImage 3 '+ mask_path +' '+ mask_path +' 0.01 1')
        os.system(self.ants_path +'ImageMath 3 '+ mask_path +' MD '+ mask_path +' 1')
        os.system(self.ants_path +'ImageMath 3 '+ mask_path +' ME '+ mask_path +' 1')
        os.system(self.ants_path +'CopyImageHeaderInformation '+ t1_path+' '+ mask_path +' '+ mask_path +' 1 1 1')
        mask = np.uint8(nib.load(mask_path).get_data())
        return mask


    def get_localization(self, t1_v, t1c_v, t2_v, flair_v, brain_mask):
        """
		ABLnetwork output, finds the brain, Whole tumor region

		t1_v       = t1 volume (numpy array)
		t1c_v      = t1c volume (numpy array)
		t2_v       = t2 volume (numpy array)
		flair_v    = flair volume (numpy array)
		brain_mask = brain, whole tumor mask (numpy array, output of ANTs pieline)
        """

        t1_v    = normalize(t1_v,    brain_mask)
        t1c_v   = normalize(t1c_v,   brain_mask)
        t2_v    = normalize(t2_v,    brain_mask)
        flair_v = normalize(flair_v, brain_mask)

        generated_output_logits = np.empty((self.ABLnclasses, flair_v.shape[0],flair_v.shape[1],flair_v.shape[2]))

        for slices in tqdm(range(flair_v.shape[2])):
            flair_slice = np.transpose(flair_v[:,:,slices])
            t2_slice    = np.transpose(t2_v[:,:,slices])
            t1ce_slice  = np.transpose(t1c_v[:,:,slices])
            t1_slice    = np.transpose(t1_v[:,:,slices])  
                          
            array        = np.zeros((flair_slice.shape[0],flair_slice.shape[1],4))
            array[:,:,0] = flair_slice
            array[:,:,1] = t2_slice
            array[:,:,2] = t1ce_slice
            array[:,:,3] = t1_slice
                

            transformed_array = torch.from_numpy(convert_image(array)).float()
            transformed_array = transformed_array.unsqueeze(0) ## neccessary if batch size == 1
            transformed_array = transformed_array.to(self.device)
            logits            = self.ABLnet(transformed_array).detach().cpu().numpy()# 3 x 240 x 240  
            generated_output_logits[:,:,:, slices] = logits.transpose(0, 1, 3, 2)

        final_pred  = apply_argmax_to_logits(generated_output_logits)
        final_pred  = perform_postprocessing(final_pred)
        final_pred  = adjust_classes_air_brain_tumour(np.uint8(final_pred))

        return np.uint8(final_pred)


    def inner_class_classification_with_logits_NCube(self, t1, 
                                                        t1ce, t2, flair, 
                                                        brain_mask, mask, N = 64):
        """
		output of 3D tiramisu model (tir3Dnet)

		mask = numpy array output of ABLnet 
		N = patch size during inference
        """

        t1    = normalize(t1, brain_mask)
        t1ce  = normalize(t1ce, brain_mask)
        t2    = normalize(t2, brain_mask)
        flair = normalize(flair, brain_mask)

        shape = t1.shape # to exclude batch_size
        final_prediction = np.zeros((self.T3Dnclasses, shape[0], shape[1], shape[2]))
        x_min, x_max, y_min, y_max, z_min, z_max = bbox(mask, pad = N)
        
        x_min, x_max, y_min, y_max, z_min, z_max = x_min, min(shape[0] - N, x_max), y_min, min(shape[1] - N, y_max), z_min, min(shape[2] - N, z_max)
        with torch.no_grad():
            for x in tqdm(range(x_min, x_max, N//2)):
                for y in range(y_min, y_max, N//2):
                    for z in range(z_min, z_max, N//2):
                        high = np.zeros((1, 4, N, N, N))

                        high[0, 0, :, :, :] = flair[x:x+N, y:y+N, z:z+N]
                        high[0, 1, :, :, :] = t2[x:x+N, y:y+N, z:z+N]
                        high[0, 2, :, :, :] = t1[x:x+N, y:y+N, z:z+N]
                        high[0, 3, :, :, :] = t1ce[x:x+N, y:y+N, z:z+N]

                        high = Variable(torch.from_numpy(high)).to(self.device).float()
                        pred = torch.nn.functional.softmax(self.Tir3Dnet(high).detach().cpu())
                        pred = pred.data.numpy()

                        final_prediction[:, x:x+N, y:y+N, z:z+N] = pred[0]

        final_prediction = convert5class_logitsto_4class(final_prediction)

        return final_prediction


    def inner_class_classification_with_logits_DualPath(self, t1, 
                                                            t1ce, t2, flair, 
                                                            brain_mask, mask=None, 
                                                            prediction_size = 9):

        """
		output of BNet3D 

		prediction_size = mid inference patch size 
        """

        t1    = normalize(t1, brain_mask)
        t1ce  = normalize(t1ce, brain_mask)
        t2    = normalize(t2, brain_mask)
        flair = normalize(flair, brain_mask)

        shape = t1.shape # to exclude batch_size
        final_prediction = np.zeros((self.B3Dnclasses, shape[0], shape[1], shape[2]))

        x_min, x_max, y_min, y_max, z_min, z_max = bbox(mask, pad = prediction_size)

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
                    pred  = torch.nn.functional.softmax(self.BNET3Dnet(high, low1, pred_size=prediction_size).detach().cpu())
                    pred  = pred.numpy()

                    final_prediction[:, x:x+prediction_size, y:y+prediction_size, z:z+prediction_size] = pred[0]

        final_prediction = convert5class_logitsto_4class(final_prediction)

        return final_prediction


    def inner_class_classification_with_logits_2D(self, 
                                                    t1ce_volume, 
                                                    t2_volume, 
                                                    flair_volume):
        """
		output of 2D tiramisu model (MNet)
		
		
        """
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        transformSequence=transforms.Compose(transformList)

        generated_output = np.empty((self.Mnclasses,flair_volume.shape[0],flair_volume.shape[1],flair_volume.shape[2]))
        for slices in tqdm(range(flair_volume.shape[2])):
            flair_slice = scale_every_slice_between_0_to_255(np.transpose(flair_volume[:,:,slices]))
            t2_slice    = scale_every_slice_between_0_to_255(np.transpose(t2_volume[:,:,slices]))
            t1ce_slice  = scale_every_slice_between_0_to_255(np.transpose(t1ce_volume[:,:,slices]))

            array        = np.zeros((flair_slice.shape[0],flair_slice.shape[1],3))
            array[:,:,0] = flair_slice
            array[:,:,1] = t2_slice
            array[:,:,2] = t1ce_slice
            array = np.uint8(array)
            transformed_array = transformSequence(array)
            transformed_array = transformed_array.unsqueeze(0)
            transformed_array = transformed_array.to(self.device)
            outs = torch.nn.functional.softmax(self.MNET2D(transformed_array).detach().cpu()).numpy()
            outs = np.swapaxes(generated_output,1, 2)

        return outs


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

        brain_mask = self.get_ants_mask(t2_path)

        mask  =  self.get_localization(t1, t1ce, t2, flair, brain_mask)
        # mask  =  np.swapaxes(mask,1, 0)
           
        if not self.quick:
            final_predictionTir3D_logits  = self.inner_class_classification_with_logits_NCube(t1, t1ce, t2, flair, brain_mask, mask)
            final_predictionBNET3D_logits = self.inner_class_classification_with_logits_DualPath(t1, t1ce, t2, flair, brain_mask, mask)
            final_predictionMnet_logits   = self.inner_class_classification_with_logits_2D(t1, t2, flair).transpose(0, 2, 1, 3)
            final_prediction_array        = np.array([final_predictionTir3D_logits, final_predictionBNET3D_logits, final_predictionMnet_logits])
        else:
            final_predictionMnet_logits   = self.inner_class_classification_with_logits_2D(t1, t2, flair)
            final_prediction_array       = np.array([final_predictionMnet_logits])

        final_prediction_logits = combine_logits_AM(final_prediction_array)
        final_pred              = postprocessing_pydensecrf(final_prediction_logits)
        final_pred              = combine_mask_prediction(mask, final_pred)
        final_pred              = perform_postprocessing(final_pred)
        final_pred              = adjust_classes(final_pred)

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            save_volume(final_pred, affine, os.path.join(save_path, 'DeepBrainSeg_Prediction'))

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
        flair =  nib.load(os.path.join(path, name + 'flair.nii.gz')).get_data()
        t1    =  nib.load(os.path.join(path, name + 't1.nii.gz')).get_data()
        t1ce  =  nib.load(os.path.join(path, name + 't1ce.nii.gz')).get_data()
        t2    =  nib.load(os.path.join(path, name + 't2.nii.gz')).get_data()
        affine=  nib.load(os.path.join(path, name + 'flair.nii.gz')).affine
        
        print ("[INFO: DeepBrainSeg] (" + strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + ") Working on: ", path)

        brain_mask =  self.get_ants_mask(os.path.join(path, name + 't2.nii.gz'))
        # brain_mask = get_brain_mask(t1)
        mask       =  self.get_localization(t1, t1ce, t2, flair, brain_mask)
        mask       =  np.swapaxes(mask,1, 0)

        if not self.quick:
            final_predictionTir3D_logits  = self.inner_class_classification_with_logits_NCube(t1, t1ce, t2, flair, brain_mask, mask)
            final_predictionBNET3D_logits = self.inner_class_classification_with_logits_DualPath(t1, t1ce, t2, flair, brain_mask, mask)
            final_predictionMnet_logits   = self.inner_class_classification_with_logits_2D(t1, t2, flair)
            final_prediction_array        = np.array([final_predictionTir3D_logits, final_predictionBNET3D_logits, final_predictionMnet_logits])
        else:
            final_predictionMnet_logits   = self.inner_class_classification_with_logits_2D(t1, t2, flair)
            final_prediction_array       = np.array([final_predictionMnet_logits])

        final_prediction_logits = combine_logits_AM(final_prediction_array)
        final_pred              = postprocessing_pydensecrf(final_prediction_logits)
        final_pred              = combine_mask_prediction(mask, final_pred)
        final_pred              = perform_postprocessing(final_pred)
        final_pred              = adjust_classes(final_pred)
        if save:
            save_volume(final_pred, affine, os.path.join(path, 'DeepBrainSeg_Prediction'))
        return final_pred



# ========================================================================================

if __name__ == '__main__':
    ext = deepSeg(True)
    ext.get_segmentation_brats('../../sample_volume/Brats18_CBICA_AVG_1/')
