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


def GenerateCSV(model, dataset_path, logs_root, iteration):
    model.eval()

    brainRegion = []; backgroundRegion = []; 
    ETRegion = []; TCRegion = []; WTRegion = []
    ETDice = []; TCDice = []; WTDice = []
    path = []; coordinate = []; 


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
                        wt, tc, et = np.sum(get_dice_score(pred, mask))

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
    dataFrame['ETRegion'] = ETRegion
    dataFrame['TCRegion'] = TCRegion
    dataFrame['WTRegion'] = WTRegion
    dataFrame['brain']    = brainRegion
    dataFrame['ETdice']  = ETDice
    dataFrame['WTdice']  = WTDice
    dataFrame['TCdice']  = TCDice
    dataFrame['background'] = backgroundRegion
    dataFrame['coordinate'] = coordinate

    save_path = os.path.join(logs_root, 'csv/iteration_{}.csv'.format(iteration))
    pd.to_csv(dataFrame, save_path)
    return save_path



class Trainer():

    def __init__(self, csv_path = None, 
                    data_root = None,
                    logs_root = None):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = "cpu"

        map_location = device

        self.T3Dnclasses = 5
        self.Tir3Dnet = FCDenseNet57(self.T3Dnclasses)
        ckpt = torch.load(ckpt_tir3D, map_location=map_location)
        self.Tir3Dnet.load_state_dict(ckpt['state_dict'])
        print ("================================== TIRNET3D Loaded =================================")
        self.Tir3Dnet = self.Tir3Dnet.to(device)

        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        self.optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-05, weight_decay=1e-5) 
        self.scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
        self.optimizer.load_state_dict(ckpt['optimizer'])
        
        
        #-------------------- SETTINGS: LOSS
        weights   = torch.FloatTensor([0.38398745, 1.48470261, 1.,         1.61940178, 0.2092336]).to(device)
        self.loss = torch.nn.CrossEntropyLoss(weight = weights)

        self.start_epoch = 0
        self.hardmine_every = 10
        self.hardmine_iteration = 0

        self.dataRoot = data_root
        self.csv_path = csv_path

        if not csv_path:
            self.csvPath = GenerateCSV(self.Tir3Dnet, self.dataRoot, logs_root, iteration = self.hardmine_iteration)

    def train(self, nnClassCount, trBatchSize, trMaxEpoch, timestampLaunch, checkpoint):

        #---- TRAIN THE NETWORK
        sub = pd.DataFrame()
        lossMIN    = 100000
        accMax     = 0
        timestamps = []
        losses = []
        accs = []
        wt_dice_scores=[]
        tc_dice_scores=[]
        et_dice_scores=[]

        for epochID in range (self.start_epoch, trMaxEpoch):

            if (epochID % self.hardmine_every) == 0:
                self.hardmine_iteration += 1
                self.csvPath = GenerateCSV(self.Tir3Dnet, self.dataRoot, logs_root, iteration = self.hardmine_iteration)

            #-------------------- SETTINGS: DATASET BUILDERS

            ## TODO: data loader
            np.random.shuffle(TrainVolPaths)
            np.random.shuffle(ValidVolPaths)
            datasetTrain = custom.Generator(imgs = TrainVolPaths[:40000])
            datasetVal =   custom.Generator(imgs = ValidVolPaths[:2500])

            dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=8, pin_memory=False)
            dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=True, num_workers=8, pin_memory=False)


            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime


            print (str(epochID)+"/" + str(trMaxEpoch) + "---")
            self.epochTrain (self.Tir3Dnet, 
                            dataLoaderTrain, 
                            self.optimizer, 
                            self.scheduler, 
                            trMaxEpoch, 
                            nnClassCount, 
                            self.loss, 
                            trBatchSize)

            lossVal, losstensor, wt_dice_score, tc_dice_score, et_dice_score, _cm = self.epochVal (self.Tir3Dnet, 
                                                                                            dataLoaderVal, 
                                                                                            self.optimizer, 
                                                                                            self.scheduler, 
                                                                                            trMaxEpoch, 
                                                                                            nnClassCount, 
                                                                                            self.loss, 
                                                                                            trBatchSize)


            currAcc = float(np.sum(np.eye(nclasses)*_cm.conf))/np.sum(_cm.conf)
            print (_cm.conf)


            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            launchTimestamp = timestampDate + '-' + timestampTime



            scheduler.step(losstensor.item())

            if lossVal < lossMIN:
                lossMIN = lossVal

                timestamps.append(launchTimestamp)
                losses.append(lossVal)
                accs.append(currAcc)
                wt_dice_scores.append(wt_dice_score)
                tc_dice_scores.append(tc_dice_score)
                et_dice_scores.append(et_dice_score)

                model_name = 'model_loss = ' + str(lossVal) + '_acc = '+str(currAcc) + '_best_loss.pth.tar'
                
                states = {'epochID': epochID + 1,
                            'state_dict': model.state_dict(),
                            'best_acc': currAcc,
                            'confusion_matrix':_cm.conf,
                            'best_loss':lossMIN,
                            'optimizer' : self.optimizer.state_dict()}

                torch.save(states, os.path.join(self.logs_root, 'models', model_name))
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + launchTimestamp + '] loss= ' + str(lossVal) + ' wt_dice_score='+str(wt_dice_score)+' tc_dice_score='+str(tc_dice_score) +' et_dice_score='+str(et_dice_score))

            elif currAcc > accMax:
                accMax  = currAcc
                timestamps.append(launchTimestamp)
                losses.append(lossVal)
                accs.append(accMax)
                wt_dice_scores.append(wt_dice_score)
                tc_dice_scores.append(tc_dice_score)
                et_dice_scores.append(et_dice_score)

                model_name = 'model_loss = ' + str(lossVal) + '_acc = '+str(currAcc) + '_best_acc.pth.tar'

                states = {'epochID': epochID + 1,
                            'state_dict': model.state_dict(),
                            'best_acc': accMax,
                            'confusion_matrix':_cm.conf,
                            'best_loss':lossVal,
                            'optimizer' : self.optimizer.state_dict()}

                torch.save(states, os.path.join(self.logs_root, 'models', model_name))
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + launchTimestamp + '] loss= ' + str(lossVal) + ' wt_dice_score='+str(wt_dice_score)+' tc_dice_score='+str(tc_dice_score) +' et_dice_score='+str(et_dice_score) + ' Acc = '+ str(currAcc))


            else:
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + launchTimestamp + '] loss= ' + str(lossVal) + ' wt_dice_score='+str(wt_dice_score)+' tc_dice_score='+str(tc_dice_score) +' et_dice_score='+str(et_dice_score))


        sub['timestamp'] = timestamps
        sub['loss'] = losses
        sub['WT_dice_score'] = wt_dice_scores
        sub['TC_dice_score'] = tc_dice_scores
        sub['ET_dice_score'] = et_dice_scores

        sub.to_csv(os.path.join(self.logs_root, 'training.csv'), index=True)


    #--------------------------------------------------------------------------------
    def epochTrain (self, model, dataLoader, optimizer, scheduler, epochMax, classCount, loss, trBatchSize):

        phase='train'
        with torch.set_grad_enabled(phase == 'train'):
            for batchID, (high_res, seg, weight_map, _) in tqdm(enumerate (dataLoader)):
                
                target = seg.long()
                high_res = high_res.float()
                # weight_map = weight_map.float() / torch.max(weight_map)

                varInputHigh = high_res.to(device)
                varTarget    = target.to(device)
                varMap       = weight_map.to(device)
                # print (varInputHigh.size(), varTarget.size())

                varOutput = model(varInputHigh)
                
                cross_entropy_lossvalue = loss(varOutput, varTarget)

                # assert False
                # cross_entropy_lossvalue = torch.mean(cross_entropy_lossvalue)
                dice_loss_ =  dice_loss(varOutput, varTarget)
                lossvalue  = cross_entropy_lossvalue + dice_loss_
                # lossvalue  = cross_entropy_lossvalue


                # print(lossvalue.size(), varOutput.size(), varMap.size())
                lossvalue = torch.mean(lossvalue)

                optimizer.zero_grad()
                lossvalue.backward()
                optimizer.step()

    #--------------------------------------------------------------------------------
    def epochVal (self, model, dataLoader, optimizer, scheduler, epochMax, classCount, loss, trBatchSize):

        model.eval ()

        lossVal = 0
        lossValNorm = 0

        losstensorMean = 0
        confusion_meter.reset()

        wt_dice_score, tc_dice_score, et_dice_score = 0.0, 0.0, 0.0
        with torch.no_grad():
            for i, (high_res, seg, weight_map, _) in enumerate (dataLoader):

                # print (_)
                target = seg.long()
                high_res = high_res.float()
                # weight_map = weight_map.float()/ torch.max(weight_map)

                varInputHigh = high_res.to(device)
                varTarget    = target.to(device)
                varMap       = weight_map.to(device)
                # print (varInputHigh.size(), varTarget.size())

                varOutput = model(varInputHigh)
                _, preds = torch.max(varOutput,1)

                wt_, tc_, et_ = self.get_dice_score(varOutput, varTarget)
                wt_dice_score += wt_
                tc_dice_score += tc_
                et_dice_score += et_

                cross_entropy_lossvalue = loss(varOutput, varTarget)

                # assert False
                # cross_entropy_lossvalue = torch.mean(cross_entropy_lossvalue)
                dice_loss_              =  dice_loss(varOutput, varTarget)

                losstensor  =  cross_entropy_lossvalue + dice_loss_
                # losstensor  =  cross_entropy_lossvalue 
                # print varOutput, varTarget
                losstensorMean += losstensor
                confusion_meter.add(preds.data.view(-1), varTarget.data.view(-1))
                lossVal += losstensor.item()
                del losstensor,_,preds
                del varOutput, varTarget, varInputHigh
                lossValNorm += 1

            wt_dice_score, tc_dice_score, et_dice_score = wt_dice_score/lossValNorm, tc_dice_score/lossValNorm, et_dice_score/lossValNorm
            outLoss = lossVal / lossValNorm
            losstensorMean = losstensorMean / lossValNorm

        return outLoss, losstensorMean, wt_dice_score, tc_dice_score, et_dice_score, confusion_meter