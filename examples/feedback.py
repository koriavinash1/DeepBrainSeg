import os
import sys
import torch
sys.path.append('..')

from os.path import expanduser
from DeepBrainSeg.tumor.feedBack import GenerateCSV
from DeepBrainSeg.tumor.models.modelTir3D import FCDenseNet57


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

home = expanduser("~")
ckpt_tir3D    = os.path.join(home, '.DeepBrainSeg/BestModels/Tramisu_3D_FC57_best_acc.pth.tar')

Tir3Dnet = FCDenseNet57(5)
ckpt = torch.load(ckpt_tir3D, map_location=device)
Tir3Dnet.load_state_dict(ckpt['state_dict'])
print ("================================== TIRNET3D Loaded =================================")
Tir3Dnet = Tir3Dnet.to(device)

GenerateCSV(Tir3Dnet, '../sample_volume/brats', '../Logs/')
