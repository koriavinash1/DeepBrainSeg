# ANTS based brain mask

import os
import numpy as np
import nibabel as nib

def get_ants_mask(t1_path, save_path, ants_path= '/opt/ANTs/bin/'):
        """
		We make use of ants framework for generalized skull stripping
		
		t1_path: t1 volume path (str)
		saves the mask in the same location as t1 data directory
		returns: maskvolume (numpy uint8 type) 
        """
        mask_path = os.path.join(os.path.dirname(t1_path), 'mask.nii.gz')
        os.system(ants_path +'ImageMath 3 '+ mask_path +' Normalize '+ t1_path)
        os.system(ants_path +'ThresholdImage 3 '+ mask_path +' '+ mask_path +' 0.01 1')
        os.system(ants_path +'ImageMath 3 '+ mask_path +' MD '+ mask_path +' 1')
        os.system(ants_path +'ImageMath 3 '+ mask_path +' ME '+ mask_path +' 1')
        os.system(ants_path +'CopyImageHeaderInformation '+ t1_path+' '+ mask_path +' '+ mask_path +' 1 1 1')
        mask = np.uint8(nib.load(mask_path).get_data())
        os.makedirs(os.path.basename(save_path), exist_ok=True)
        nib_obj = nib.load(t1_path)
        vol = nib_obj.get_data()
        affine = nib_obj.affine
        print (save_path)
        volume = np.uint8(vol*mask)
        volume = nib.Nifti1Image(volume, affine)
        volume.set_data_dtype(np.uint8)
        nib.save(volume, save_path)
        
        return mask
