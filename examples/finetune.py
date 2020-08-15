import os
import sys
import time
sys.path.append('..')

from DeepBrainSeg.tumor import FineTuner 

finetune = FineTuner('../../../../Logs/csv/training.csv',
                    '../../../../Logs/csv/validation.csv',
                    '../../../../MICCAI_BraTS2020_TrainingData',
                    '../../../../Logs')

ckpt_path = '../../../../Logs/models/model_loss = 0.2870774023637236_acc = 0.904420656270211_best_loss.pth.tar'
timestampTime = time.strftime("%H%M%S")
timestampDate = time.strftime("%d%m%Y")
timestampLaunch = timestampDate + '-' + timestampTime
size = 64
# finetune.train(nnClassCount = nclasses,    
#               trBatchSize = 4, 
#               trMaxEpoch = 50, 
#               timestampLaunch = timestampLaunch, 
#               checkpoint = ckpt_path)

validation_set = '../../../../MICCAI_BraTS2020_ValidationData'
save_path = '../../../../First_Validation_Set_Size_{}'.format(size)
finetune.infer(ckpt_path, validation_set, save_path, size)