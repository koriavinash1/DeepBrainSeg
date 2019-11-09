import numpy as np
import os
import sys
sys.path.append('..')
from DeepBrainSeg.helpers import imshow
from DeepBrainSeg import tumorSeg

path = '../sample_volume/Brats18_CBICA_AME_1'
segmentor = tumorSeg(quick=True)
segmentor.get_segmentation_brats(path)
