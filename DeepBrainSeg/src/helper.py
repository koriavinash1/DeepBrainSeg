from deepbrain import Extractor
from scipy.ndimage import uniform_filter, maximum_filter
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
import SimpleITK as sitk
import numpy as np
import nibabel as nib

from scipy.ndimage.measurements import label
from skimage.morphology import erosion, dilation

#=======================================================================================
# init. brain mask extractor
ext = Extractor()
def get_brain_mask(t1):
    probs = ext.run(t1)
    return probs > 0.5

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
