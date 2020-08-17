import numpy as np
import os
import sys
sys.path.append('..')

from DeepBrainSeg.tumor import tumorSeg

path = '../sample_volume/brats/Brats18_2013_3_1'
segmentor = tumorSeg(quick=False)
segmentor.get_segmentation_brats(path)
