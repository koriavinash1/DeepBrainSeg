from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__license__ = 'MIT'
__version__ = '0.1'
__maintainer__ = ['Avinash Kori']
__email__ = ['koriavinash1@gmail.com']


from .src.Tester import deepSeg
from .src.helper import *
import os
from time import gmtime, strftime
from google_drive_downloader import GoogleDriveDownloader as gdd

model_path = '/tmp/DeepBrainSeg/BestModels'
if (not os.path.exists(model_path)) or (os.listdir(model_path) == []):
	os.makedirs(model_path)
	print ("[INFO: DeepBrainSeg] (" + strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + ") " + 'Tramisu_3D_FC57_best_acc.pth.tar')
	gdd.download_file_from_google_drive(file_id='1FM0zMBZ1Njdc63KfZ9jDc9qzib6d5A7v',
                                    dest_path=os.path.join(model_path, 'Tramisu_3D_FC57_best_acc.pth.tar'))

	print ("[INFO: DeepBrainSeg] (" + strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + ") " + 'Tramisu_2D_FC57_best_loss.pth.tar')
	gdd.download_file_from_google_drive(file_id='1JrTIV65ZTyPTMNswYXRF5R92xFYaV1y6',
                                    dest_path=os.path.join(model_path, 'Tramisu_2D_FC57_best_loss.pth.tar'))

	print ("[INFO: DeepBrainSeg] (" + strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + ") " + 'BrainNet_3D_best_acc.pth.tar')
	gdd.download_file_from_google_drive(file_id='1Fu13If4JNEqTbD8x8JHA74aoQxVYv-Ma',
                                    dest_path=os.path.join(model_path, 'BrainNet_3D_best_acc.pth.tar'))

	print ("[INFO: DeepBrainSeg] (" + strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + ") " + 'ABL_CE_best_model_loss_based.pth.tar')
	gdd.download_file_from_google_drive(file_id='1HUbx3xXFvGU2eQ66N7vcaCLqNChIekxM',
                                    dest_path=os.path.join(model_path, 'ABL_CE_best_model_loss_based.pth.tar'))
else :
	print ("[INFO: DeepBrainSeg] (" + strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + ") " + 'Skipping Download Files already exists')