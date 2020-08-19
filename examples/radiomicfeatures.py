import numpy as np
import os
import sys
sys.path.append('..')

from DeepBrainSeg.tumor import tumorSeg
from DeepBrainSeg.tumor.dataGenerator import nii_loader
from DeepBrainSeg.radiomics import ExtractRadiomicFeatures

path = '../sample_volume/brats/Brats18_2013_11_1'
segmentor = tumorSeg(quick=True)
prediction = segmentor.get_segmentation_brats(path)


spath = {
	't1': os.path.join(path, path.split('/')[-1] + '_t1.nii.gz'),
	't2': os.path.join(path, path.split('/')[-1] + '_t2.nii.gz'),
	# 'flair': os.path.join(path, path.split('/')[-1] + '_flair.nii.gz'),
	't1ce': os.path.join(path, path.split('/')[-1] + '_t1ce.nii.gz')
}
vol, _, _ = nii_loader(spath)

infoclasses = {}
infoclasses['whole'] = (1,2,4,)
infoclasses['ET'] = (4,)
infoclasses['CT'] = (1,4,)


for seq in vol.keys():
	for key in infoclasses.keys():
		mask = np.zeros_like(prediction)
		for iclass in infoclasses[key]:
			mask[prediction == iclass] = 1

		radfeatures = ExtractRadiomicFeatures(vol[seq],
									mask,
									save_path = os.path.join(path, 
													'DBSRadFeatures', 
													'Seq_{}_Class_{}'.format(seq, key)),
									seq = seq,
									class_ = key)
		df = radfeatures.all_features()
