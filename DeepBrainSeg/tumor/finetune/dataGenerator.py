import torch.utils.data as data
import nibabel as nib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.morphology as morph
from skimage.feature import canny
import matplotlib.pyplot as plt
from cv2 import bilateralFilter
from skimage.restoration import denoise_tv_chambolle
from skimage.restoration import denoise_tv_bregman
from skimage.transform import resize
import SimpleITK as sitk
import torch

def nii_loader(path, coordinate = (0,0,0), size = 64):
    #"""
    #Now given a path, we have to load
    #1) High res version of flair,t2,t1 and t1c
    #2) Low  res version of flair,t2,t1 and t1c
    #3) If the path comprises of "Non" then make an empty 9x9x9 array
    #4) if not, load the segmentation mask too.


    #Accepts
    #1) path of data

    #Returns
    #1) a 4D high resolution data
    #2) a 4d low resolution data
    #3) Segmentation
    #"""

    flair = nib.load(os.path.join(path, 'flair.nii.gz')).get_data()
    t2 = nib.load(os.path.join(path, 't2.nii.gz')).get_data()
    t1 = nib.load(os.path.join(path, 't1.nii.gz')).get_data()
    t1ce = nib.load(os.path.join(path, 't1ce.nii.gz')).get_data()
    try:
        brainmask = nib.load(os.path.join(path, 'mask.nii.gz')).get_data()
    else:
        brainmask = get_ants_mask(ants_path, os.path.join(path, 't1.nii.gz'))

    sege_mask = np.uint8(nib.load(path+'/'+'seg.nii.gz').get_data())
    sege_mask[np.where(sege_mask==4)]=3
    sege_mask[np.where(sege_mask==5)]=4  ## making an effort to make classes 0,1,2,3,4 rather than 0,1,2,4,5


    data = np.zeros((4, size, size, size))

    try:
        shape = flair.shape
        data[0,:,:,:] = flair[coordinate[0]:coordinate[0] + size,
				coordinate[1]:coordinate[1] + size,
				coordinate[2]:coordinate[2] + size]
        data[1,:,:,:] = t2[coordinate[0]:coordinate[0] + size,
				coordinate[1]:coordinate[1] + size,
				coordinate[2]:coordinate[2] + size]
        data[2,:,:,:] = t1[coordinate[0]:coordinate[0] + size,
				coordinate[1]:coordinate[1] + size,
				coordinate[2]:coordinate[2] + size]
        data[3,:,:,:] = t1ce[coordinate[0]:coordinate[0] + size,
				coordinate[1]:coordinate[1] + size,
				coordinate[2]:coordinate[2] + size]
    except:
        shape = flair_high_res.shape
        print(path, shape)
    return high_resolution_data, sege_mask


def multilabel_binarize(image_nD, nlabel):
    labels = range(nlabel)
    out_shape = (len(labels),) + image_nD.shape
    bin_img_stack = np.ones(out_shape, dtype='uint8')
    for label in labels:
        bin_img_stack[label] = np.where(image_nD == label, bin_img_stack[label], 0)
    return bin_img_stack



selem = morph.disk(1)
def getEdge(label):
    out = np.float32(np.zeros(label.shape))
    for i in range(label.shape[0]):
        out[i,:,:] = morph.binary_dilation(canny(np.float32(label[i,:,:])), selem)
    
    return out


def getEdgeEnhancedWeightMap_3D(label, label_ids =[0,1,2,3,4], scale=1, edgescale=1, assign_equal_wt=False):
    
    label = multilabel_binarize(label, len(label_ids))# convert to onehot vector
    # print (np.unique(label), label.shape)
    shape = (0,)+label.shape[1:]
    weight_map = np.empty(shape, dtype='uint8')
    if assign_equal_wt:
        return np.ones_like(label)
    for i in range(label.shape[0]): 
        #Estimate weight maps:
        weights = np.ones(len(label_ids))
        slice_map = np.ones(label[i,:,:,:].shape)
        for _id in label_ids:
            class_frequency = np.sum(label[i,:,:,:] == label_ids[_id])
            if class_frequency:
                weights[label_ids.index(_id)] = scale*label[i,:,:,:].size/class_frequency
                slice_map[np.where(label[i,:,:,:]==label_ids.index(_id))] = weights[label_ids.index(_id)]
                edge = getEdge(label[i,:,:,:])
                edge_frequency = np.sum(np.sum(edge==1.0))
                if edge_frequency:    
                    slice_map[np.where(edge==1.0)] += edgescale*label[i,:,:,:].size/edge_frequency
            # print (weights)
            # utils.imshow(edge, cmap='gray')
        # utils.imshow(weight_map, cmap='gray')
        weight_map = np.append(weight_map, np.expand_dims(slice_map, axis=0), axis=0)
    return np.sum(np.float32(weight_map), 0)



class ImageFolder(data.Dataset):
    """
    """

    def __init__(self, imgs, 
		 transform=None, 
		 target_transform=None,
                 loader=nii_loader):


        # imgs = make_dataset(root) ### root= entire csv path
        self.imgs = imgs#[:16]
        np.random.shuffle(self.imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path= self.imgs[index]
        high_res_input, segmentation = self.loader(path)
        high_res_input= torch.from_numpy(high_res_input).float() ## torchifying the data.
        

        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        weight_map = getEdgeEnhancedWeightMap_3D(segmentation)
        return high_res_input, segmentation, weight_map, path

    def __len__(self):
        return len(self.imgs)


def getDataPaths(path):
    data = pd.read_csv(path)
    imgpaths = data['Paths'].as_matrix()
    np.random.shuffle(imgpaths)
    return imgpaths


if __name__== '__main__':

    csv='./training_patch_info.csv'
    a= getDataPaths(csv)

    dl=ImageFolder(a)
    for i, (x,z,_) in enumerate (dl):
        print ('min',np.min(z),'max',np.max(z), _)
        # print('size',z.long().size())
