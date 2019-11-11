# HD-BET based brain mask
# https://github.com/MIC-DKFZ/HD-BET

import os
import numpy as np
import nibabel as nib
import sys

def get_bet_mask(vol_path, save_path, device = 0):
        """
		We make use of ants framework for generalized skull stripping
		
		t1_path: t1 volume path (str)
		saves the mask in the same location as t1 data directory
		returns: maskvolume (numpy uint8 type) 
        """
        print (save_path)
        filename = vol_path.split('/').pop().split('.')[0]
        os.system('hd-bet -i '+ vol_path + ' -device '+ str(device) + ' -mode fast -tta 0')
        os.makedirs(save_path, exist_ok=True)
        os.system('mv ' + os.path.join(os.path.dirname(vol_path), filename+'_bet.nii.gz') + ' ' +  os.path.join(save_path, filename+'.nii.gz')) 
        os.system('mv ' + os.path.join(os.path.dirname(vol_path), filename+'_bet_mask.nii.gz') + ' ' + os.path.join(save_path, filename+'_mask.nii.gz')) 
        volume = np.uint8(nib.load(os.path.join(save_path, filename+'_mask.nii.gz')).get_data())
        return volume
