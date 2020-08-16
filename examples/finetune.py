import os
import sys
import time
import torch

sys.path.append('..')
from DeepBrainSeg.tumor import FineTuner 
from DeepBrainSeg.tumor.feedBack import GenerateCSV3D 
from DeepBrainSeg.tumor.dataGenerator import Generator
from DeepBrainSeg.tumor.models.modelTir3D import FCDenseNet57

from os.path import expanduser
home = expanduser("~")
base_ckpt_path = os.path.join(home, '.DeepBrainSeg/BestModels/Tramisu_3D_FC57_best_acc.pth.tar')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nclasses = 5
model = FCDenseNet57(nclasses)
ckpt = torch.load(base_ckpt_path, map_location=device)
model.load_state_dict(ckpt['state_dict'])


finetune = FineTuner(model,
					nclasses,
					'../Logs',
					device,
					antehoc_feedback = GenerateCSV3D,
					gradual_unfreeze = True)



# ckpt_path = '../../Logs/models/model_loss = 0.280981784279302_acc = 0.9003648718656386_best_loss.pth.tar'
finetune.train('../Logs/csv/training.csv',
              '../Logs/csv/validation.csv',
              '../sample_volume/brats',
              trBatchSize = 4, 
              trMaxEpoch = 50, 
              DataGenerator = Generator)



# ckpt_path = '../../Logs/models/model_loss = 0.280981784279302_acc = 0.9003648718656386_best_loss.pth.tar'
size = 64
validation_set = '../sample_volume/valid'
save_path = '../First_Validation_Set_Size_{}'.format(size)
finetune.infer(ckpt_path, validation_set, save_path, size)
