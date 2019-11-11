import os, sys
import numpy as np
import dicom2nifti
from time import gmtime, strftime
import dicom2nifti.settings as settings

def singleDicom2nifti(input_path, output_path, verbose=False):
    """
    """
    if not os.path.exists(input_path): raise ValueError("Path doesn't exist")
    
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if verbose:
        print ("[INFO: DeepBrainSeg] (" + strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + ") " + 'Working on: {}'.format(input_path))

    dicom2nifti.dicom_series_to_nifti(input_path, output_path)


def convertDcm2nifti(path_json, output_dir, verbose=False):
    """
        path_json: {'key1': path1, 'key2': path2}
        output_dir: nifty save dir path
    """

    for key in path_json.keys():
        input_path = path_json[key]
        output_path = os.path.join(output_dir, key + '.nii.gz')
        singleDicom2nifti(input_path, output_path, verbose)
    

