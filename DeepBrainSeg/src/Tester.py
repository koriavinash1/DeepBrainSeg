import torch
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from torch.autograd import Variable
from skimage.transform import resize
from torchvision import transforms

from tqdm import tqdm
import pdb
import os

from helper import *


#========================================================================================
# prediction functions.....................


class DeepBrainSeg():
    def __init__(self, quick=False):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = "cpu"

        map_location = device

        #========================================================================================

        ckpt_tir2D    = '/tmp/DeepBrainSeg/BestModels/Tramisu_2D_FC57_best_loss.pth.tar'
        ckpt_tir3D    = '/tmp/DeepBrainSeg/BestModels/Tramisu_3D_FC57_best_acc.pth.tar'
        ckpt_BNET3D   = '/tmp/DeepBrainSeg/BestModels/BrainNet_3D_best_acc.pth.tar'
        ckpt_ABL      = '/tmp/DeepBrainSeg/BestModels/ABL_CE_best_model_loss_based.pth.tar'

        #========================================================================================
        # air brain lesion segmentation..............
        import modelABL as ABL

        ABLnclasses = 3
        self.ABLnet = ABL.FCDenseNet103(n_classes = ABLnclasses) ## intialize the graph
        saved_parms=torch.load(ckpt_ABL, map_location=map_location) 
        ABLnet.load_state_dict(saved_parms['state_dict']) ## fill the model with trained params
        print ("=================================== ABLNET2D Loaded =================================")
        self.ABLnet.eval()
        self.ABLnet = ABLnet.to(device)

        #========================================================================================
        # Tir2D net.......................
        import modelTir2D as mnet
        Mnclasses = 4
        self.mnet = mnet.FCDenseNet57(Mnclasses)
        ckpt = torch.load(ckpt_tir2D, map_location=map_location)
        mnet.load_state_dict(ckpt['state_dict'])
        print ("=================================== MNET2D Loaded ===================================")
        self.mnet.eval()
        self.mnet = mnet.to(device)

        #========================================================================================

        if not quick:
            # BrainNet3D model......................
            import model3DBNET as Bnet3D
            B3Dnclasses = 5
            self.BNET3Dnet = Bnet3D.BrainNet_3D_Inception()
            ckpt = torch.load(ckpt_BNET3D, map_location=map_location)
            BNET3Dnet.load_state_dict(ckpt['state_dict'])
            print ("=================================== KAMNET3D Loaded =================================")
            self.BNET3Dnet.eval()
            self.BNET3Dnet = BNET3Dnet.to(device)

            #========================================================================================
            # Tir3D model...................
            import modelTir3D as Tir3D

            T3Dnclasses = 5
            self.Tir3Dnet = Tir3D.FCDenseNet57(T3Dnclasses)
            ckpt = torch.load(ckpt_tir3D, map_location=map_location)
            Tir3Dnet.load_state_dict(ckpt['state_dict'])
            print ("================================== TIRNET2D Loaded =================================")
            self.Tir3Dnet.eval()
            self.Tir3Dnet = Tir3Dnet.to(device)


        #========================================================================================

        self.device = device
        self.quick = quick


    def get_localization(self, t1_v, t1c_v, t2_v, flair_v, brain_mask, mask_net):
        """
        """

        mask_net.eval()
        mask_net.to(self.device)

        t1_v    = normalize(t1_v,    brain_mask)
        t1c_v   = normalize(t1c_v,   brain_mask)
        t2_v    = normalize(t2_v,    brain_mask)
        flair_v = normalize(flair_v, brain_mask)

        generated_output_logits = np.empty((3, flair_v.shape[0],flair_v.shape[1],flair_v.shape[2]))

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
            transformed_array = transformed_array.to(device)
            logits            = mask_net(transformed_array).detach().cpu().numpy()# 3 x 240 x 240  
            
            generated_output_logits[:,:,:, slices] = logits

        final_pred  = apply_argmax_to_logits(generated_output_logits)
        final_pred  = perform_postprocessing(final_pred)
        final_pred  = adjust_classes_air_brain_tumour(np.uint8(final_pred))

        del mask_net
        return np.uint8(final_pred)


    def inner_class_classification_with_logits_NCube(self, t1, 
                                                        t1ce, t2, flair, 
                                                        brain_mask, mask, 
                                                        model, N = 64):
        """
        """
        model.eval()
        model.to(self.device)

        t1    = normalize(t1, brain_mask)
        t1ce  = normalize(t1ce, brain_mask)
        t2    = normalize(t2, brain_mask)
        flair = normalize(flair, brain_mask)

        shape = t1.shape # to exclude batch_size
        final_prediction = np.zeros((T3Dnclasses, shape[0], shape[1], shape[2]))
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

                        high = Variable(torch.from_numpy(high)).to(device).float()
                        pred = torch.nn.functional.softmax(model(high).detach().cpu())
                        pred = pred.data.numpy()

                        final_prediction[:, x:x+N, y:y+N, z:z+N] = pred[0]
        del model
        final_prediction = convert5class_logitsto_4class(final_prediction)

        return final_prediction


    def inner_class_classification_with_logits_DualPath(self, t1, 
                                                            t1ce, t2, flair, 
                                                            brain_mask, mask=None, 
                                                            model=None, prediction_size = 9):

        """
        """
        model.eval()
        model.to(self.device)

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

                    high = Variable(torch.from_numpy(high)).to(device).float()
                    low1  = Variable(torch.from_numpy(low1)).to(device).float()
                    pred = torch.nn.functional.softmax(model(high, low1, pred_size=prediction_size).detach().cpu()).numpy()

                    final_prediction[:, x:x+prediction_size, y:y+prediction_size, z:z+prediction_size] = pred[0]

        del model 
        final_prediction = convert5class_logitsto_4class(final_prediction)

        return final_prediction


    def inner_class_classification_with_logits_2D(self, t1ce_volume, t2_volume, flair_volume, model):
        """
        """
        model.eval()
        model.to(self.device)
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        transformSequence=transforms.Compose(transformList)

        generated_output = np.empty((Mnclasses,flair_volume.shape[0],flair_volume.shape[1],flair_volume.shape[2]))
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
            transformed_array = transformed_array.to(device)
            outs = torch.nn.functional.softmax(model(transformed_array).detach().cpu()).numpy()
            outs = np.swapaxes(generated_output,1, 2)

        del model

        return outs


    def get_brainsegmentation(t1, t2, t1ce, flair):
        """
        """
        brain_mask = get_brain_mask(t1)

        mask  =  get_localization(t1, t1ce, t2, flair, brain_mask, ABLnet)
        mask  =  np.swapaxes(mask,1, 0)
           
        # final_predictionTir3D_logits = inner_class_classification_with_logits_64Cube(t1, t1ce, t2, flair, brain_mask, mask, Tir3Dnet)
        # final_predictionBNET3D_logits = inner_class_classification_with_logits_DualPath(t1, t1ce, t2, flair, brain_mask, mask, BNET3Dnet)
        final_predictionMnet_logits  = inner_class_classification_with_logits_2D(t1, t2, flair, mnet)

        #final_prediction_array  = np.array([final_predictionTir3D_logits, final_predictionBNET3D_logits, final_predictionMnet_logits])
        final_prediction_array  = np.array([final_predictionMnet_logits])
        final_prediction_logits = combine_logits_AM(final_prediction_array)
        final_pred              = postprocessing_pydensecrf(final_prediction_logits)
        final_pred              = combine_mask_prediction(mask, final_pred)
        final_pred              = perform_postprocessing(final_pred)
        final_pred              = adjust_classes(final_pred)

        return final_pred




    def get_segmentation(self, t1, t2, t1ce, flair):
        """
        """
        brain_mask = get_brain_mask(t1)

        mask  =  get_localization(t1, t1ce, t2, flair, brain_mask, ABLnet)
        mask  =  np.swapaxes(mask,1, 0)
           
        final_predictionTir3D_logits = inner_class_classification_with_logits_64Cube(t1, t1ce, t2, flair, brain_mask, mask, Tir3Dnet)
        final_predictionBNET3D_logits = inner_class_classification_with_logits_DualPath(t1, t1ce, t2, flair, brain_mask, mask, BNET3Dnet)
        final_predictionMnet_logits  = inner_class_classification_with_logits_2D(t1, t2, flair, mnet)

        final_prediction_array  = np.array([final_predictionTir3D_logits, final_predictionBNET3D_logits, final_predictionMnet_logits])
        #final_prediction_array = np.array([final_predictionMnet_logits])
        final_prediction_logits = combine_logits_AM(final_prediction_array)
        final_pred              = postprocessing_pydensecrf(final_prediction_logits)
        final_pred              = combine_mask_prediction(mask, final_pred)
        final_pred              = perform_postprocessing(final_pred)
        final_pred              = adjust_classes(final_pred)

        return final_pred

    def get_segmentation_brats(self, path, save):
        """
        """

        flair =  nib.load(path + 'flair.nii.gz').get_data()
        t1    =  nib.load(path + 't1.nii.gz').get_data()
        t1ce  =  nib.load(path + 't1ce.nii.gz').get_data()
        t2    =  nib.load(path + 't2.nii.gz').get_data()
        affine=  nib.load(path + 'flair.nii.gz').affine
        
        brain_mask = get_brain_mask(t1)
        print ("[INFO: DeepBrainSeg] Working on: ", path)
        
        mask  =  get_localization(t1, t1ce, t2, flair, brain_mask, ABLnet)
        mask  =  np.swapaxes(mask,1, 0)
        
        final_predictionTir3D_logits = inner_class_classification_with_logits_64Cube(t1, t1ce, t2, flair, brain_mask, mask, Tir3Dnet)
        final_predictionBNET3D_logits = inner_class_classification_with_logits_DualPath(t1, t1ce, t2, flair, brain_mask, mask, BNET3Dnet)
        final_predictionMnet_logits  = inner_class_classification_with_logits_2D(t1, t2, flair, mnet)


        final_prediction_array  = np.array([final_predictionTir3D_logits, final_predictionBNET3D_logits, final_predictionMnet_logits])
        final_prediction_logits = combine_logits_AM(final_prediction_array)
        final_pred              = postprocessing_pydensecrf(final_prediction_logits)
        final_pred              = combine_mask_prediction(mask, final_pred)
        final_pred              = perform_postprocessing(final_pred)
        final_pred              = adjust_classes(final_pred)
        save_volume(final_pred, affine, root_path +'Prediction') 
        return 1



# ========================================================================================

if __name__ == '__main__':
    submit = True


    if not submit:
        # Predictions..............
        names     =  os.listdir('./data')

        for name in names:
            root_path = './data/'+name+'/'
            path = root_path + name +'_'

            seg   =  np.uint8(nib.load(path+'seg.nii.gz').get_data())

            flair =  nib.load(path + 'flair.nii.gz').get_data()
            t1    =  nib.load(path + 't1.nii.gz').get_data()
            t1ce  =  nib.load(path + 't1ce.nii.gz').get_data()
            t2    =  nib.load(path + 't2.nii.gz').get_data()
            affine=  nib.load(path + 'flair.nii.gz').affine
            
            try:
                brain_mask = np.uint8(nib.load(root_path + 'mask.nii.gz').get_data())
            except:
                generate_mask()
                brain_mask = np.uint8(nib.load(root_path + 'mask.nii.gz').get_data())
            print ("Working on: ", path)
     
            mask  =  get_localization(t1, t1ce, t2, flair, brain_mask, ABLnet)
            mask  =  np.swapaxes(mask,1, 0)
           
            final_predictionTir3D_logits = inner_class_classification_with_logits_64Cube(t1, t1ce, t2, flair, brain_mask, mask, Tir3Dnet)
            final_predictionBNET3D_logits = inner_class_classification_with_logits_DualPath(t1, t1ce, t2, flair, brain_mask, mask, BNET3Dnet)
            final_predictionMnet_logits  = inner_class_classification_with_logits_2D(t1, t2, flair, mnet)


            final_prediction_array  = np.array([final_predictionTir3D_logits, final_predictionBNET3D_logits, final_predictionMnet_logits])
            final_prediction_logits = combine_logits_AM(final_prediction_array)
            final_pred              = postprocessing_pydensecrf(final_prediction_logits)
            final_pred              = combine_mask_prediction(mask, final_pred)
            final_pred              = perform_postprocessing(final_pred)
            final_pred              = adjust_classes(final_pred)
            wt, tc, et              = get_dice_score(final_pred, seg)
            print ('Whole Tumour DiceScore = '+ str(wt) +'; Tumour Core DiceScore = '+ str(tc) +'; Enhancing Tumour DiceScore = '+str(et))
            save_volume(final_pred, affine, root_path +'Prediction') 

    else :
        names     =  os.listdir('./data/Test')
        root_save = './data/Results/'
        os.makedirs(root_save, exist_ok=True)

        for name in names:
            root_path = './data/Test/'+name+'/'
            path = root_path + name + '_'

            flair =  nib.load(path + 'flair.nii.gz').get_data()
            t1    =  nib.load(path + 't1.nii.gz').get_data()
            t1ce  =  nib.load(path + 't1ce.nii.gz').get_data()
            t2    =  nib.load(path + 't2.nii.gz').get_data()
            affine=  nib.load(path + 'flair.nii.gz').affine

            try:
                brain_mask = np.uint8(nib.load(root_path + 'mask.nii.gz').get_data())
            except:
                generate_mask()
                brain_mask = np.uint8(nib.load(root_path + 'mask.nii.gz').get_data())

            print ("Working on: ", path)

            mask  =  get_localization(t1, t1ce, t2, flair, brain_mask, ABLnet)
            mask  =  np.swapaxes(mask,1, 0)


            final_predictionTir3D_logits = inner_class_classification_with_logits_64Cube(t1, t1ce, t2, flair, brain_mask, mask, Tir3Dnet)
            final_predictionKam3D_logits = inner_class_classification_with_logits_DualPath(t1, t1ce, t2, flair, brain_mask, mask, Kam3Dnet)
            final_predictionMnet_logits  = inner_class_classification_with_logits_2D(t1, t2, flair, mnet)

            final_prediction_array  = np.array([final_predictionTir3D_logits, final_predictionKam3D_logits, final_predictionMnet_logits])
            final_prediction_logits = combine_logits_AM(final_prediction_array)

            final_pred              = postprocessing_pydensecrf(final_prediction_logits)
            final_pred              = combine_mask_prediction(mask, final_pred)
            final_pred              = perform_postprocessing(final_pred)
            final_pred              = adjust_classes(final_pred)
            save_volume(final_pred, affine, root_save + name)   
