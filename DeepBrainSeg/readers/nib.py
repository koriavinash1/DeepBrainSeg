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

import os
import tempfile
from time import time
import datetime
import numpy as np
import nibabel as nib

class nib_loader(object):
    """
    """
    def __init__(self):
        pass

    def load_vol(self, path):
        """
            path : patient data path

            returns numpy array of patient data
        """
        self.patient = nib.load(path)
        self.affine  = self.patient.affine

        return self.patient.get_data()

    def write_vol(self, path, volume):
        """
            path : path to write the data
            vol : modifient volume

            return: True or False based on saving of volume
        """
        try:
            volume = np.uint8(volume)
            volume = nib.Nifti1Image(volume, self.affine)
            volume.set_data_dtype(np.uint8)
            nib.save(volume, path)
            return True
        except: 
            return False


