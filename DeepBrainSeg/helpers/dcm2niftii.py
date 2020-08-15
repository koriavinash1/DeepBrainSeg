#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# author: Avinash Kori
# contact: koriavinash1@gmail.com
# MIT License

# Copyright (c) 2020 Avinash Kori

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


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
    

