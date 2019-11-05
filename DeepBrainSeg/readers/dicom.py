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


