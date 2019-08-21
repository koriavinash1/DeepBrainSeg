import torch
import SimpleITK as sitk
import numpy as np
from scipy.ndimage.measurements import label
from skimage.morphology import erosion, dilation
import nibabel as nib
from torch.autograd import Variable
from skimage.transform import resize
from torchvision import transforms

from tqdm import tqdm
import pdb
import os
from scipy.ndimage import uniform_filter, maximum_filter
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#========================================================================================

ckpt_tir2D = './BestModels/Tramisu_2D_FC57_best_loss.pth.tar'
ckpt_tir3D    = './BestModels/Tramisu_3D_FC57_best_acc.pth.tar'
ckpt_BNET3D   = './BestModels/BrainNet_3D_best_acc.pth.tar'
ckpt_ABL      = './BestModels/ABL_CE_best_model_loss_based.pth.tar'

#========================================================================================
# Tir3D model...................
import modelTir3D as Tir3D

T3Dnclasses = 5
Tir3Dnet = Tir3D.FCDenseNet57(T3Dnclasses)
ckpt = torch.load(ckpt_tir3D)
Tir3Dnet.load_state_dict(ckpt['state_dict'])
print ("================================== TIRNET2D Loaded =================================")
Tir3Dnet.eval()
Tir3Dnet = Tir3Dnet.to(device)

#========================================================================================
# air brain lesion segmentation..............
import modelABL as ABL

ABLnclasses = 3
ABLnet = ABL.FCDenseNet103(n_classes = ABLnclasses) ## intialize the graph
saved_parms=torch.load(ckpt_ABL) 
ABLnet.load_state_dict(saved_parms['state_dict']) ## fill the model with trained params
print ("=================================== ABLNET2D Loaded =================================")
ABLnet = ABLnet.to(device)
ABLnet.eval()

#========================================================================================
# Tir2D net.......................
import modelTir2D as mnet
Mnclasses = 4
mnet = mnet.FCDenseNet57(Mnclasses)
ckpt = torch.load(ckpt_magicnet)
mnet.load_state_dict(ckpt['state_dict'])
print ("=================================== MNET2D Loaded ===================================")
mnet.eval()
mnet = mnet.to(device)

#========================================================================================
# BrainNet3D model......................
import model3DBNET as Bnet3D
B3Dnclasses = 5
BNET3Dnet = Bnet3D.BrainNet_3D_Inception()
ckpt = torch.load(ckpt_BNET3D)
BNET3Dnet.load_state_dict(ckpt['state_dict'])
print ("=================================== KAMNET3D Loaded =================================")
BNET3Dnet.eval()
BNET3Dnet = BNET3Dnet.to(device)

#========================================================================================

# Helper Functions.....................

def bbox(vol, pad = 18):
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


def perform_postprocessing(voxels, threshold=12000):
    c,n = label(voxels)
    nums = np.array([np.sum(c==i) for i in range(1, n+1)])
    selected_components = nums>threshold
    selected_components[np.argmax(nums)] = True
    mask = np.zeros_like(voxels)
    # print(selected_components.tolist())
    for i,select in enumerate(selected_components):
        if select:
            mask[c==(i+1)]=1
    return mask*voxels


def normalize(img,mask):
    mean=np.mean(img[mask!=0])
    std=np.std(img[mask!=0])
    return (img-mean)/std


def adjust_classes_air_brain_tumour(volume):
    ""
    volume = np.uint8(volume)
    volume[volume == 1] = 0
    volume[volume == 2] = 1
    return volume


def convert_image(image):
    ""
    x= np.float32(image)
    x=np.swapaxes(x,0,2)
    x= np.swapaxes(x,1,2)
    return x


def apply_argmax_to_logits(logits):
    "logits dimensions: nclasses, height, width, depth"
    logits = np.argmax(logits, axis=0)         
    return np.uint8(logits)


def adjust_classes(volume):
    ""
    volume = np.uint8(volume)
    volume[volume == 4] = 0
    volume[volume == 3] = 4
    return volume


def save_volume(volume, affine, path):
    volume = np.uint8(volume)
    volume = nib.Nifti1Image(volume, affine)
    volume.set_data_dtype(np.uint8)
    nib.save(volume, path +'.nii.gz')
    pass


def scale_every_slice_between_0_to_255(a):
    normalized_a=  255*((a-np.min(a))/(np.max(a)-np.min(a)))
    return normalized_a
    

def class_wise_postprocessing(logits):
    "logits dimension: nclasses, width, height, depth"
    return_ = np.zeros_like(logits)
    for class_ in range(logits.shape[0]):
        return_[class_, :, :, :] = perform_postprocessing(logits[class_, :, :, :])

    return return_

def get_whole_tumor(data):
    return (data>0)*(data<4)


def get_tumor_core(data):
    return np.logical_or(data==1,data==4)


def get_enhancing_tumor(data):
    return data==4


def get_dice_score(prediction, ground_truth):
    # print (np.unique(prediction), np.unique(ground_truth))
    masks=(get_whole_tumor, get_tumor_core, get_enhancing_tumor)
    p    =np.uint8(prediction)
    gt   =np.uint8(ground_truth)
    wt,tc,et=[2*np.sum(func(p)*func(gt)) / (np.sum(func(p)) + np.sum(func(gt))+1e-6) for func in masks]
    return wt, tc, et


def postprocessing_pydensecrf(logits):
    # probs of shape 3d image per class: Nb_classes x Height x Width x Depth
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

def convert5class_logitsto_4class(logits):
    assert len(logits.shape) == 4
    assert logits.shape[0] == 5

    new_logits = np.zeros((4,)+ logits.shape[1:])
    new_logits[0, :, :, :] = logits[0,:,:,:] + logits[4,:,:,:]
    new_logits[1, :, :, :] = logits[1,:,:,:]
    new_logits[2, :, :, :] = logits[2,:,:,:]
    new_logits[3, :, :, :] = logits[3,:,:,:]
    return new_logits

def combine_logits_GM(x):
    pdb.set_trace()
    # x array of logits
    assert len(x.shape) == 5
    final = np.ones_like(x[0], dtype='float32')
    for ii in x:
        final = final*ii*10.
    return  (final**(1./x.shape[0]))/10.


def combine_logits_AM(x):
    # x array of logits
    assert len(x.shape) == 5
    final = np.zeros_like(x[0])
    for ii in x:
        final = final + ii
    return final * (1/len(x))


def combine_predictions_GM(x):
    # x array of logits
    x = np.array(x, dtype='float32')
    assert len(x.shape) == 4
    final = np.ones_like(x[0])
    for ii in x:
        final = final*ii
    return np.uint8(final**(1.0/len(x)))


def combine_predictions_AM(x):
    # x array of logits
    x = np.array(x, dtype='float32')
    assert len(x.shape) == 4
    final = np.zeros_like(x[0])
    for ii in x:
        final = final + ii
    return np.uint8(final * (1.0/len(x)))

def combine_mask_prediction(mask, pred):
    mask[mask == 1]   = 2
    mask[pred == 1]   = 1
    mask[pred == 3]   = 3
    return mask

#========================================================================================
# prediction functions.....................

def get_localization(t1_v, t1c_v, t2_v, flair_v, brain_mask, mask_net):
    mask_net.eval()
    mask_net.to(device)

    t1_v = normalize(t1_v, brain_mask)
    t1c_v = normalize(t1c_v, brain_mask)
    t2_v = normalize(t2_v, brain_mask)
    flair_v = normalize(flair_v, brain_mask)

    generated_output_logits = np.empty((3, flair_v.shape[0],flair_v.shape[1],flair_v.shape[2]))

    for slices in tqdm(range(flair_v.shape[2])):
        flair_slice= np.transpose(flair_v[:,:,slices])
        t2_slice= np.transpose(t2_v[:,:,slices])
        t1ce_slice= np.transpose(t1c_v[:,:,slices])
        t1_slice= np.transpose(t1_v[:,:,slices])  
                      
        array=np.zeros((flair_slice.shape[0],flair_slice.shape[1],4))
        array[:,:,0]=flair_slice
        array[:,:,1]=t2_slice
        array[:,:,2]=t1ce_slice
        array[:,:,3]=t1_slice
            

        transformed_array = torch.from_numpy(convert_image(array)).float()
        transformed_array=transformed_array.unsqueeze(0) ## neccessary if batch size == 1
        transformed_array= transformed_array.to(device)
        logits = mask_net(transformed_array).detach().cpu().numpy()# 3 x 240 x 240  
        
        generated_output_logits[:,:,:, slices] = logits

    final_pred  = apply_argmax_to_logits(generated_output_logits)
    final_pred  = perform_postprocessing(final_pred)
    final_pred  = adjust_classes_air_brain_tumour(np.uint8(final_pred))

    del mask_net
    return np.uint8(final_pred)


def inner_class_classification_with_logits_64Cube(t1, t1ce, t2, flair, brain_mask, mask, model):
    model.eval()
    model.to(device)

    t1 = normalize(t1, brain_mask)
    t1ce = normalize(t1ce, brain_mask)
    t2 = normalize(t2, brain_mask)
    flair = normalize(flair, brain_mask)

    shape = t1.shape # to exclude batch_size
    final_prediction = np.zeros((T3Dnclasses, shape[0], shape[1], shape[2]))
    x_min, x_max, y_min, y_max, z_min, z_max = bbox(mask, pad = 64)
    
    x_min, x_max, y_min, y_max, z_min, z_max = x_min, min(shape[0] - 64, x_max), y_min, min(shape[1] - 64, y_max), z_min, min(shape[2] - 64, z_max)
    with torch.no_grad():
        for x in tqdm(range(x_min, x_max, 32)):
            for y in range(y_min, y_max, 32):
                for z in range(z_min, z_max, 32):
                    high = np.zeros((1, 4, 64, 64, 64))

                    high[0, 0, :, :, :] = flair[x:x+64, y:y+64, z:z+64]
                    high[0, 1, :, :, :] = t2[x:x+64, y:y+64, z:z+64]
                    high[0, 2, :, :, :] = t1[x:x+64, y:y+64, z:z+64]
                    high[0, 3, :, :, :] = t1ce[x:x+64, y:y+64, z:z+64]

                    high = Variable(torch.from_numpy(high)).to(device).float()
                    pred = torch.nn.functional.softmax(model(high).detach().cpu())
                    pred = pred.data.numpy()

                    final_prediction[:, x:x+64, y:y+64, z:z+64] = pred[0]
    del model
    final_prediction = convert5class_logitsto_4class(final_prediction)

    return final_prediction


def inner_class_classification_with_logits_DualPath(t1, t1ce, t2, flair, brain_mask, mask=None, model=None, prediction_size = 9):
    model.eval()
    model.to(device)

    t1 = normalize(t1, brain_mask)
    t1ce = normalize(t1ce, brain_mask)
    t2 = normalize(t2, brain_mask)
    flair = normalize(flair, brain_mask)

    shape = t1.shape # to exclude batch_size
    final_prediction = np.zeros((K3Dnclasses, shape[0], shape[1], shape[2]))
    # x_min, x_max, y_min, y_max, z_min, z_max = 0, shape[0], 0, shape[1], 0, shape[2]
    
    # if mask.any() != None:
    x_min, x_max, y_min, y_max, z_min, z_max = bbox(mask, pad = 9)

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
                low[0, 0], low[0, 1], low[0, 2], low[0, 3] = low[0, 0] + flair[0,0,0], low[0, 1] + t2[0,0,0], low[0, 2] + t1[0,0,0], low[0, 2] + t1ce[0,0,0]
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


def inner_class_classification_with_logits_2D(t1ce_volume, t2_volume, flair_volume, model):
    model.eval()
    model.to(device)
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
        array       = np.zeros((flair_slice.shape[0],flair_slice.shape[1],3))
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

#========================================================================================
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
