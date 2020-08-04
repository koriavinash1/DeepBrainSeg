from glob import glob
import panda as pd
import numpy as np
from dataGenerator import nii_loader, get_patch
from ..models.modelTir3D import FCDenseNet57
from ../..helpers.helper import *


def GenerateCSV(model, dataset_path, logs_root, iteration):
	model.eval()

	brainRegion = []; C1Region = []; C2Region = []; C3Region = []
	backgroundRegion = []; path = []; coordinate = []; dice = []

	def _GenerateSegmentation_(spath, vol, seg, size = 64, nclasses = 5):
        """
		output of 3D tiramisu model (tir3Dnet)

		mask = numpy array output of ABLnet 
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
                    	eDice = np.sum(get_dice_score(pred, mask))

                    	coordinate.append((x, y, z))
                    	path.append(spath)
                    	backgroundRegion.append(np.mean(mask == 0))
                    	C1Region.append(np.mean(mask == 1))
                    	C2Region.append(np.mean(mask == 2))
                    	C3Region.append(np.mean(mask == 3))
                    	brainRegion.append(np.mean(mask == 4))
                    	dice.append(eDice)

        final_prediction = convert5class_logitsto_4class(final_prediction)

        return final_prediction


    subjects = os.listdir(dataset_path)
    spath = {}
    for subject in subjects:
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
	dataFrame['c1'] = C1Region
	dataFrame['c2'] = C2Region
	dataFrame['c3'] = C3Region
	dataFrame['brain'] = brainRegion
	dataFrame['dice']  = dice
	dataFrame['background'] = backgroundRegion
	dataFrame['coordinate'] = coordinate

	save_path = os.path.join(logs_root, 'csv/iteration_{}.csv'.format(iteration))
	pd.to_csv(dataFrame, save_path)
	return save_path



class Training():

	def __init__(self, csv_path = None, 
					data_root = None,
					logs_root = None):
        self.T3Dnclasses = 5
        self.Tir3Dnet = FCDenseNet57(self.T3Dnclasses)
        ckpt = torch.load(ckpt_tir3D, map_location=map_location)
        self.Tir3Dnet.load_state_dict(ckpt['state_dict'])
        print ("================================== TIRNET3D Loaded =================================")
        self.Tir3Dnet = self.Tir3Dnet.to(device)

        self.dataRoot = data_root
        self.csv_path = csv_path

        if not csv_path:
        	self.csvPath = GenerateCSV(self.Tir3Dnet, dataRoot, logs_root, iteration = 0)

    def train(self):
    	pass

