from glob import glob
import panda as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.autograd import Variable
from torchvision import transforms

from dataGenerator import nii_loader, get_patch, Generator
from ..models.modelTir3D import FCDenseNet57
from ../..helpers.helper import *


def __get_whole_tumor__(data):
    return (data > 0)*(data < 4)

def __get_tumor_core__(data):
    return np.logical_or(data == 1, data == 3)

def __get_enhancing_tumor__(data):
    return data == 3

def _get_dice_score_(prediction, ground_truth):

    masks = (__get_whole_tumor__, __get_tumor_core__, __get_enhancing_tumor__)
    p     = np.uint8(prediction)
    gt    = np.uint8(ground_truth)
    wt, tc, et=[2*np.sum(func(p)*func(gt)) / (np.sum(func(p)) + np.sum(func(gt))+1e-6) for func in masks]
    return wt, tc, et


def GenerateCSV(model, dataset_path, logs_root):
    model.eval()

    brainRegion = []; backgroundRegion = []; 
    ETRegion = []; TCRegion = []; WTRegion = []
    ETDice = []; TCDice = []; WTDice = []
    path = []; coordinate = []; 


    def _GenerateSegmentation_(spath, vol, seg, size = 64, nclasses = 5):
        """
        output of 3D tiramisu model (tir3Dnet)

        N = patch size during inference
        """

        shape = vol['t1'].shape # to exclude batch_size
        final_prediction = np.zeros((nclasses, shape[0], shape[1], shape[2]))
        x_min, x_max, y_min, y_max, z_min, z_max = 0, shape[0], 0, shape[1], 0, shape[2]
        x_min, x_max, y_min, y_max, z_min, z_max = x_min, min(shape[0] - size, x_max), y_min, min(shape[1] - size, y_max), z_min, min(shape[2] - size, z_max)

        with torch.no_grad():
            for x in tqdm(range(x_min, x_max, size//2)):
                for y in range(y_min, y_max, size//2):
                    for z in range(z_min, z_max, size//2):

                        data, mask = get_patch(vol, seg, coordinate = (x, y, z), size = size)
                        data = Variable(torch.from_numpy(data)).to(self.device).float()
                        pred = torch.nn.functional.softmax(model(data).detach().cpu())
                        pred = pred.data.numpy()

                        final_prediction[:, x:x + size, y:y + size, z:z + size] = pred[0]

                        # Logs update
                        wt, tc, et = _get_dice_score_(pred, mask)

                        coordinate.append((x, y, z))
                        path.append(spath)
                        backgroundRegion.append(np.mean(mask == 0))
                        WTRegion.append(np.mean(__get_whole_tumor__(mask)))
                        ETRegion.append(np.mean(__get_enhancing_tumor__(mask)))
                        TCRegion.append(np.mean(__get_tumor_core__(mask))
                        brainRegion.append(np.mean(mask == 4)))
                        ETDice.append(et); WTDice.append(wt); TCDice.append(tc)

        final_prediction = convert5class_logitsto_4class(final_prediction)

        return final_prediction


    subjects = os.listdir(dataset_path)
    training_subjects = subjects[:int(.8*len(subjects))]
    validation_subjects = subjects[int(.8*len(subjects)):]

    for i, subjects in enumerate([training_subjects, validation_subjects]):
        for subject in subjects:
            spath = {}
            subject_path = os.path.join(dataset_path, subject)
            spath['flair'] = os.path.join(subject_path, subject + '_flair.nii.gz')
            spath['t1ce']  = os.path.join(subject_path, subject + '_t1ce.nii.gz')
            spath['seg']   = os.path.join(subject_path, subject + '_seg.nii.gz')
            spath['t1']    = os.path.join(subject_path, subject + '_t1.nii.gz')
            spath['t2']    = os.path.join(subject_path, subject + '_t2.nii.gz')
            spath['mask']  = os.path.join(dataset_path, 'mask.nii.gz')

            vol, seg, affine = nii_loader(spath)
            predictions = _GenerateSegmentation_(subject_path, vol, seg, size = 64, nclasses = 5)
            save_volume(predictions, affine, os.path.join(subject_path, 'DeepBrainSeg_Prediction'))


        dataFrame = pd.DataFrame()
        dataFrame['path'] = path
        dataFrame['ETRegion'] = ETRegion
        dataFrame['TCRegion'] = TCRegion
        dataFrame['WTRegion'] = WTRegion
        dataFrame['brain']    = brainRegion
        dataFrame['ETdice']  = ETDice
        dataFrame['WTdice']  = WTDice
        dataFrame['TCdice']  = TCDice
        dataFrame['background'] = backgroundRegion
        dataFrame['coordinate'] = coordinate

        if i == 0: save_path = os.path.join(logs_root, 'csv/training.csv')
        else: save_path = os.path.join(logs_root, 'csv/validation.csv')

        os.makedirs(save_path, exist_ok=True)
        pd.to_csv(dataFrame, save_path)
    return save_path



if __name__ == '__main__':
    T3Dnclasses = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    map_location = device
    Tir3Dnet = FCDenseNet57(T3Dnclasses)
    ckpt = torch.load(ckpt_tir3D, map_location=map_location)
    Tir3Dnet.load_state_dict(ckpt['state_dict'])
    print ("================================== TIRNET3D Loaded =================================")
    Tir3Dnet = Tir3Dnet.to(device)

    GenerateCSV(Tir3Dnet, '../', '../Logs/')