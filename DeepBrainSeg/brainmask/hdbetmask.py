# HD-BET based brain mask
# https://github.com/MIC-DKFZ/HD-BET

import os
import numpy as np
import nibabel as nib
import sys
import subprocess


def get_bet_mask(vol_path, device = 0):
    """
    We make use of bet framework for generalized skull stripping
    
    vol_path: t1 volume path (str)
    saves the mask in the same location as t1 data directory
    returns: maskvolume (numpy uint8 type) 
    """
    mask_path = os.path.join(os.path.dirname(vol_path), 'mask.nii.gz')
    filename = vol_path.split('/').pop().split('.')[0]
    command = 'hd-bet -i '+ vol_path + ' -device '+ str(device) + ' -mode fast -tta 0'
    os.system("{} > /dev/null".format(command))
    try:
        os.system('mv ' + os.path.join(os.path.dirname(vol_path), filename +'_bet_mask.nii.gz') + ' ' + mask_path) 
    except:
        print ("Mask Already exists")

    os.system('rm ' + os.path.join(os.path.dirname(vol_path), filename +'_bet.nii.gz')) 
    mask = np.uint8(nib.load(mask_path).get_data())
    return mask

def bet_skull_stripping(t1_path, save_path):
    """
    We make use of bet framework for generalized skull stripping
    
    t1_path: t1 volume path (str)
    saves the mask in the same location as t1 data directory
    returns: maskvolume (numpy uint8 type) 
    """
    mask = get_bet_mask(t1_path)

    os.makedirs(os.path.basename(save_path), exist_ok=True)
    nib_obj = nib.load(t1_path)
    vol = nib_obj.get_data()
    affine = nib_obj.affine
    volume = np.uint8(vol*mask)
    volume = nib.Nifti1Image(volume, affine)
    volume.set_data_dtype(np.uint8)
    nib.save(volume, save_path)
    return volume
