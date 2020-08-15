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
import pydicom
import datetime
from pydicom.dataset import Dataset, FileDataset
import numpy as np


class dcm_loader(object):
    """
    """
    def __init__(self):
        pass

    def load_vol(self, path):
        """
            path : patient data path

            returns numpy array of patient data
        """
        self.patient = pydicom.dcmread(path)

        return self.patient.pixel_array

    def write_vol(self, path, vol):
        """
            path : path to write the data
            vol : modifient volume

            return: True or False based on saving of volume
        """
        suffix = '.dcm'
        filename_little_endian = tempfile.NamedTemporaryFile(suffix=suffix).name
        filename_big_endian = tempfile.NamedTemporaryFile(suffix=suffix).name
        file_meta = Dataset()
        ds = FileDataset(filename_little_endian, {},
                 file_meta=file_meta, preamble=b"\0" * 128)
        ds.PatientName = self.patient.PatientName
        ds.PatientID = self.patient.PatientID
        ds.is_little_endian = self.patient.is_little_endian
        ds.is_implicit_VR = self.patient.is_implicit_VR

        # Set creation date/time
        dt = datetime.datetime.now()
        ds.ContentDate = dt.strftime('%Y%m%d')
        timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
        ds.ContentTime = timeStr
        ds.PixelData = vol.tostring()
        try:
            ds.save_as(filename_little_endian)
            return True
        except: 
            return False


