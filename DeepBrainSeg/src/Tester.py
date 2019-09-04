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

from .helper import *


#========================================================================================
# prediction functions.....................

class deepSeg():
    """
    """
    def __init__(self, quick=False):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = "cpu"

        map_location = device

        #========================================================================================

        ckpt_tir2D    = '/tmp/DeepBrainSeg/BestModels/Tramisu_2D_FC57_best_loss.pth.tar'
        ckpt_tir3D    = '/tmp/DeepBrainSeg/BestModels/Tramisu_3D_FC57_best_acc.pth.tar'
        ckpt_BNET3D   = '/tmp/DeepBrainSeg/BestModels/BrainNet_3D_best_acc.pth.tar'
        ckpt_ABL      = '/tmp/DeepBrainSeg/BestModels/ABL_CE_best_model_loss_based.pth.tar'

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


    def get_localization(self, t1_v, t1c_v, t2_v, flair_v, brain_mask):
        """
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
            
            generated_output_logits[:,:,:, slices] = logits

        final_pred  = apply_argmax_to_logits(generated_output_logits)
        final_pred  = perform_postprocessing(final_pred)
        final_pred  = adjust_classes_air_brain_tumour(np.uint8(final_pred))

        return np.uint8(final_pred)


    def inner_class_classification_with_logits_NCube(self, t1, 
                                                        t1ce, t2, flair, 
                                                        brain_mask, mask, N = 64):
        """
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
        """

        t1    = normalize(t1, brain_mask)
        t1ce  = normalize(t1ce, brain_mask)
        t2    = normalize(t2, brain_mask)
        flair = normalize(flair, brain_mask)

        shape = t1.shape # to exclude batch_size
        final_prediction = np.zeros((K3Dnclasses, shape[0], shape[1], shape[2]))

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

                    low1  = Variable(torch.from_numpy(low1)).to(self.device).float()
                    pred = torch.nn.functional.softmax(self.BNET3Dnet(high, low1, pred_size=prediction_size).detach().cpu()).numpy()

                    final_prediction[:, x:x+prediction_size, y:y+prediction_size, z:z+prediction_size] = pred[0]

        final_prediction = convert5class_logitsto_4class(final_prediction)

        return final_prediction


    def inner_class_classification_with_logits_2D(self, t1ce_volume, t2_volume, flair_volume):
        """
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
            transformed_array =transformed_array.unsqueeze(0)
            transformed_array = transformed_array.to(self.device)
            outs = torch.nn.functional.softmax(self.MNET2D(transformed_array).detach().cpu()).numpy()
            outs = np.swapaxes(generated_output,1, 2)

        return outs


    def get_segmentation(self, t1, t2, t1ce, flair):
        """
        """
        brain_mask = get_brain_mask(t1)

        mask  =  self.get_localization(t1, t1ce, t2, flair, brain_mask)
        mask  =  np.swapaxes(mask,1, 0)
           
        if not self.quick:
            final_predictionTir3D_logits  = self.inner_class_classification_with_logits_NCube(t1, t1ce, t2, flair, brain_mask, mask)
            final_predictionBNET3D_logits = self.inner_class_classification_with_logits_DualPath(t1, t1ce, t2, flair, brain_mask, mask)
            final_predictionMnet_logits   = self.inner_class_classification_with_logits_2D(t1, t2, flair)
            final_prediction_array        = np.array([final_predictionTir3D_logits, final_predictionBNET3D_logits, final_predictionMnet_logits])
        else:
            final_predictionMnet_logits  = self.inner_class_classification_with_logits_2D(t1, t2, flair)
            final_prediction_array       = np.array([final_predictionMnet_logits])

        final_prediction_logits = combine_logits_AM(final_prediction_array)
        final_pred              = postprocessing_pydensecrf(final_prediction_logits)
        final_pred              = combine_mask_prediction(mask, final_pred)
        final_pred              = perform_postprocessing(final_pred)
        final_pred              = adjust_classes(final_pred)

        return final_pred


    def get_segmentation_brats(self, path, save):
        """
        """

        name  = path.split("/")[-1] + "_"
        flair =  nib.load(os.path.join(path, name + 'flair.nii.gz')).get_data()
        t1    =  nib.load(os.path.join(path, name + 't1.nii.gz')).get_data()
        t1ce  =  nib.load(os.path.join(path, name + 't1ce.nii.gz')).get_data()
        t2    =  nib.load(os.path.join(path, name + 't2.nii.gz')).get_data()
        affine=  nib.load(os.path.join(path, name + 'flair.nii.gz')).affine
        
        print ("[INFO: DeepBrainSeg] + (" + strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + ") Working on: ", path)
        
        if not self.quick:
            brain_mask = get_brain_mask(t1)
            mask       =  self.get_localization(t1, t1ce, t2, flair, brain_mask)
            mask       =  np.swapaxes(mask,1, 0)
            final_predictionTir3D_logits  = self.inner_class_classification_with_logits_NCube(t1, t1ce, t2, flair, brain_mask, mask)
            final_predictionBNET3D_logits = self.inner_class_classification_with_logits_DualPath(t1, t1ce, t2, flair, brain_mask, mask)
            final_predictionMnet_logits   = self.inner_class_classification_with_logits_2D(t1, t2, flair)
            final_prediction_array        = np.array([final_predictionTir3D_logits, final_predictionBNET3D_logits, final_predictionMnet_logits])
        else:
            final_predictionMnet_logits  = self.inner_class_classification_with_logits_2D(t1, t2, flair)
            final_prediction_array       = np.array([final_predictionMnet_logits])

        final_prediction_logits = combine_logits_AM(final_prediction_array)
        final_pred              = postprocessing_pydensecrf(final_prediction_logits)
        final_pred              = combine_mask_prediction(mask, final_pred)
        final_pred              = perform_postprocessing(final_pred)
        final_pred              = adjust_classes(final_pred)

        if save:
            save_volume(final_pred, affine, os.path.join(path, 'Prediction'))
        return final_pred



# ========================================================================================

if __name__ == '__main__':
    ext = deepSeg(True)
    ext.get_segmentation_brats('../../sample_volume/Brats18_CBICA_AVG_1/')