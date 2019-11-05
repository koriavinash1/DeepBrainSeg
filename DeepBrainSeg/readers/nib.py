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

        return self.patient.get_data()

    def write_vol(self, path, vol):
        """
            path : path to write the data
            vol : modifient volume

            return: True or False based on saving of volume
        """
        try:
            ds.save_as(filename_little_endian)
            return True
        except: 
            return False


