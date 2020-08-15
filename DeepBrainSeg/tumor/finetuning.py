#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# author: Avinash Kori
# contact: koriavinash1@gmail.com
# MIT License

# Copyright (c) 2020 Avinash Kori

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from torch.autograd import Variable
import torch.nn.functional as F
import torchnet as tnt
import pandas as pd
import random

from tqdm import tqdm

from .dataGenerator import Generator
from .models.modelTir3D import FCDenseNet57
from .feedBack import GenerateCSV
from .dataGenerator import nii_loader, get_patch
from ..helpers import utils
from ..helpers import postprocessing


from os.path import expanduser
home = expanduser("~")

def __get_whole_tumor__(data):
    return (data > 0)*(data < 4)

def __get_tumor_core__(data):
    return np.logical_or(data == 1, data == 3)

def __get_enhancing_tumor__(data):
    return data == 3

def _get_dice_score_(prediction, ground_truth):

    masks = (__get_whole_tumor__, __get_tumor_core__, __get_enhancing_tumor__)
    pred  = torch.exp(prediction)
    p     = np.uint8(np.argmax(pred.data.cpu().numpy(), axis=1))
    gt    = np.uint8(ground_truth.data.cpu().numpy())
    wt, tc, et = [2*np.sum(func(p)*func(gt))/ (np.sum(func(p)) + np.sum(func(gt))+1e-6) for func in masks]
    return wt, tc, et



def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor  = y.data if isinstance(y, Variable) else y
    y_tensor  = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims    = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    y_one_hot = y_one_hot.transpose(-1, 1).transpose(-1, 2)#.transpose(-1, 3) 
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot




class FineTuner():

    def __init__(self, Traincsv_path = None, 
                    Validcsv_path = None,
                    data_root = None,
                    logs_root = None,
                    nclasses = 5,
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    antehoc_feedback = GenerateCSV,
                    gradual_unfreeze = True):

        # device = "cpu"
        self.confusion_meter = tnt.meter.ConfusionMeter(nclasses, normalized=True)
        self.device = device

        map_location = device

        self.T3Dnclasses = nclasses
        self.Tir3Dnet = FCDenseNet57(self.T3Dnclasses)
        ckpt_tir3D    = os.path.join(home, '.DeepBrainSeg/BestModels/Tramisu_3D_FC57_best_acc.pth.tar')
        ckpt = torch.load(ckpt_tir3D, map_location=map_location)
        self.Tir3Dnet.load_state_dict(ckpt['state_dict'])
        print ("================================== TIRNET3D Loaded =================================")
        self.Tir3Dnet = self.Tir3Dnet.to(device)

        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        self.optimizer = optim.Adam (self.Tir3Dnet.parameters(), 
                                     lr=0.0001, betas=(0.9, 0.999), eps=1e-05, weight_decay=1e-5) 
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor = 0.1, patience = 5, mode = 'min')
        self.optimizer.load_state_dict(ckpt['optimizer'])
        
        
        #-------------------- SETTINGS: LOSS
        weights   = torch.FloatTensor([0.38398745, 1.48470261, 1.,         1.61940178, 0.2092336]).to(device)
        self.loss = torch.nn.CrossEntropyLoss(weight = weights)

        self.start_epoch = 0
        self.hardmine_every = 8
        self.hardmine_iteration = 1
        self.logs_root = logs_root
        self.dataRoot = data_root
        self.antehoc_feedback = antehoc_feedback
        self.Traincsv_path = Traincsv_path
        self.Validcsv_path = Validcsv_path
        self.gradual_unfreeze = gradual_unfreeze


    def dice_loss(self, input, target):
        """
        input is a torch variable of size BatchxnclassesxHxW 
        representing log probabilities for each class
        target is of the groundtruth, 
        shoud have same size as the input
        """

        target = to_one_hot(target, n_dims=nclasses).to(self.device)

        assert input.size() == target.size(), "Input sizes must be equal."
        assert input.dim() == 5, "Input must be a 4D Tensor."

        probs = F.softmax(input)

        num   = (probs*target).sum() + 1e-3
        den   = probs.sum() + target.sum() + 1e-3
        dice  = 2.*(num/den)
        return 1. - dice


    def train(self, nnClassCount, 
                    trBatchSize, 
                    DataGenerator, 
                    trMaxEpoch, 
                    timestampLaunch, 
                    checkpoint):

        #---- TRAIN THE NETWORK
        sub = pd.DataFrame()
        lossMIN    = 100000
        accMax     = 0


        #---- Load checkpoint
        if checkpoint != None:
            saved_parms=torch.load(checkpoint)
            self.Tir3Dnet.load_state_dict(saved_parms['state_dict'])
            # self.optimizer.load_state_dict(saved_parms['optimizer'])
            self.start_epoch= saved_parms['epochID']
            lossMIN    = saved_parms['best_loss']
            accMax     = saved_parms['best_acc']
            print (saved_parms['confusion_matrix'])

        #---- TRAIN THE NETWORK

        accs = []
        losses = []
        timestamps = []
        tc_dice_scores = []
        wt_dice_scores = []
        et_dice_scores = []

        for epochID in range (self.start_epoch, trMaxEpoch):

            if (epochID % self.hardmine_every) == (self.hardmine_every -1):
                self.Traincsv_path = self.antehoc_feedback(self.Tir3Dnet, 
                                                   self.dataRoot, 
                                                   self.logs_root, 
                                                   iteration = self.hardmine_iteration)
                self.hardmine_iteration += 1

            #-------------------- SETTINGS: DATASET BUILDERS

            datasetTrain = DataGenerator(csv_path = self.Traincsv_path,
                                                batch_size = trBatchSize,
                                                hardmine_every = self.hardmine_every,
                                                iteration = (1 + epochID) % self.hardmine_every)
            datasetVal  =   DataGenerator(csv_path = self.Validcsv_path,
                                                batch_size = trBatchSize,
                                                hardmine_every = self.hardmine_every,
                                                iteration = 0)

            dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=1, shuffle=True,  num_workers=8, pin_memory=False)
            dataLoaderVal  = DataLoader(dataset=datasetVal, batch_size=1, shuffle=True, num_workers=8, pin_memory=False)

            if self.gradual_unfreeze: 
                # Need to include this in call back, prevent optimizer reset at every epoch
                # TODO:
                self._gradual_unfreezing_(epochID % self.hardmine_every)
                self.optimizer = optim.Adam (filter(lambda p: p.requires_grad, 
                                                    self.Tir3Dnet.parameters()), 
                                                    lr=0.0001, betas=(0.9, 0.999), 
                                                    eps=1e-05, weight_decay=1e-5) 
                self.scheduler = ReduceLROnPlateau(self.optimizer, factor = 0.1, 
                                                     patience = 5, mode = 'min')
            

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime


            print (str(epochID)+"/" + str(trMaxEpoch) + "---")
            self.epochTrain (self.Tir3Dnet, 
                            dataLoaderTrain, 
                            self.optimizer, 
                            self.loss)

            lossVal, losstensor, wt_dice_score, tc_dice_score, et_dice_score, _cm = self.epochVal (self.Tir3Dnet, 
                                                                                            dataLoaderVal, 
                                                                                            self.loss)


            currAcc = float(np.sum(np.eye(nclasses)*_cm.conf))/np.sum(_cm.conf)
            print (_cm.conf)


            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            launchTimestamp = timestampDate + '-' + timestampTime



            self.scheduler.step(losstensor.item())

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
                            'state_dict': self.Tir3Dnet.state_dict(),
                            'best_acc': currAcc,
                            'confusion_matrix':_cm.conf,
                            'best_loss':lossMIN,
                            'optimizer' : self.optimizer.state_dict()}

                os.makedirs(os.path.join(self.logs_root, 'models'), exist_ok=True)
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
                            'state_dict': self.Tir3Dnet.state_dict(),
                            'best_acc': accMax,
                            'confusion_matrix':_cm.conf,
                            'best_loss':lossVal,
                            'optimizer' : self.optimizer.state_dict()}

                os.makedirs(os.path.join(self.logs_root, 'models'), exist_ok=True)
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
    def _gradual_unfreezing_(self, epochID):
        nlayers = 0
        for _ in self.Tir3Dnet.named_children(): nlayers += 1

        layer_epoch = 2*nlayers//self.hardmine_every

        for i, (name, child) in enumerate(self.Tir3Dnet.named_children()):

            if i >= nlayers - (epochID + 1)*layer_epoch:
                print(name + ' is unfrozen')
                for param in child.parameters():
                    param.requires_grad = True
            else:
                print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False    



    #--------------------------------------------------------------------------------
    def epochTrain (self, model, 
                        dataLoader, 
                        optimizer, 
                        loss):

        phase='train'
        with torch.set_grad_enabled(phase == 'train'):
            for batchID, (data, seg, weight_map) in tqdm(enumerate (dataLoader)):
                
                target = torch.cat(seg).long().squeeze(0)
                data = torch.cat(data).float().squeeze(0)
                # weight_map = torch.cat(weight_map).float().squeeze(0) / torch.max(weight_map)

                varInput  = data.to(self.device)
                varTarget = target.to(self.device)
                # varMap    = weight_map.to(self.device)

                varOutput = model(varInput)
                
                cross_entropy_lossvalue = loss(varOutput, varTarget)
                # cross_entropy_lossvalue = torch.mean(cross_entropy_lossvalue)
                dice_loss_ =  self.dice_loss(varOutput, varTarget)
                lossvalue  = cross_entropy_lossvalue + dice_loss_
                lossvalue = torch.mean(lossvalue)

                optimizer.zero_grad()
                lossvalue.backward()
                optimizer.step()

    #--------------------------------------------------------------------------------
    def epochVal (self, model, dataLoader, loss):

        model.eval ()

        lossVal = 0
        lossValNorm = 0

        losstensorMean = 0
        confusion_meter.reset()

        wt_dice_score, tc_dice_score, et_dice_score = 0.0, 0.0, 0.0
        with torch.no_grad():
            for i, (data, seg, weight_map) in enumerate(dataLoader):
                
                target = torch.cat(seg).long().squeeze(0)
                data = torch.cat(data).float().squeeze(0)
                # weight_map = torch.cat(weight_map).float().squeeze(0) / torch.max(weight_map)

                varInput  = data.to(self.device)
                varTarget = target.to(self.device)
                # varMap    = weight_map.to(self.device)

                varOutput = model(varInput)
                _, preds = torch.max(varOutput,1)

                wt_, tc_, et_ = _get_dice_score_(varOutput, varTarget)
                wt_dice_score += wt_
                tc_dice_score += tc_
                et_dice_score += et_

                cross_entropy_lossvalue = loss(varOutput, varTarget)
                # cross_entropy_lossvalue = torch.mean(cross_entropy_lossvalue)
                dice_loss_              =  self.dice_loss(varOutput, varTarget)

                losstensor  =  cross_entropy_lossvalue + dice_loss_

                losstensorMean += losstensor
                confusion_meter.add(preds.data.view(-1), varTarget.data.view(-1))
                lossVal += losstensor.item()
                del losstensor, _, preds
                del varOutput, varTarget, varInput
                lossValNorm += 1

            wt_dice_score, tc_dice_score, et_dice_score = wt_dice_score/lossValNorm, tc_dice_score/lossValNorm, et_dice_score/lossValNorm
            outLoss = lossVal / lossValNorm
            losstensorMean = losstensorMean / lossValNorm

        return outLoss, losstensorMean, wt_dice_score, tc_dice_score, et_dice_score, confusion_meter


    #--------------------------------------------------------------------------------
    def infer (self, ckpt, rootpath, save_path, size = 64):


        os.makedirs(save_path, exist_ok = True)
        saved_parms = torch.load(ckpt)
        self.Tir3Dnet.load_state_dict(saved_parms['state_dict'])
        def __get_logits__(vol):
            for key in vol.keys():
                vol[key] = np.pad(vol[key], ((size//4, size//4), (size//4, size//4), (size//4, size//4))) 
            shape = vol['t1'].shape
            final_prediction = np.zeros((self.T3Dnclasses, shape[0], shape[1], shape[2]))
            x_min, x_max = 0, shape[0] - size
            y_min, y_max = 0, shape[1] - size
            z_min, z_max = 0, shape[2] - size

            s = size//4
            with torch.no_grad():
                for x in tqdm(range(x_min, x_max, size//2)):
                    for y in range(y_min, y_max, size//2):
                        for z in range(z_min, z_max, size//2):

                            data = get_patch(vol, coordinate = (x, y, z), size = size)
                            data = Variable(torch.from_numpy(data).unsqueeze(0)).to(self.device).float()
                            pred = torch.nn.functional.softmax(self.Tir3Dnet(data).detach().cpu())
                            pred = pred.data.numpy()
                            final_prediction[:, x + s:x + 3*s, 
                                            y + s:y + 3*s, 
                                            z + s:z + 3*s] = pred[0][:, s:-s, s:-s, s:-s]
            return final_prediction[:, s:-s, s:-s, s:-s]



        for subject in tqdm(os.listdir(rootpath)):
            spath = {}
            subject_path = os.path.join(rootpath, subject)
            spath['flair'] = os.path.join(subject_path, subject + '_flair.nii.gz')
            spath['t1ce']  = os.path.join(subject_path, subject + '_t1ce.nii.gz')
            spath['seg']   = os.path.join(subject_path, subject + '_seg.nii.gz')
            spath['t1']    = os.path.join(subject_path, subject + '_t1.nii.gz')
            spath['t2']    = os.path.join(subject_path, subject + '_t2.nii.gz')
            spath['mask']  = os.path.join(subject_path, 'mask.nii.gz')

            vol, _, affine = nii_loader(spath)
            logits = __get_logits__(vol)
            final_prediction_logits = utils.convert5class_logitsto_4class(logits)
            final_pred = postprocessing.densecrf(final_prediction_logits)
            final_pred = postprocessing.connected_components(final_pred)
            final_pred = utils.adjust_classes(final_pred)

            # save final_prediction
            if isinstance(save_path, str):
                path = os.path.join(save_path, subject )
            else:
                path = os.path.join(rootpath, subject, 'DeepBrainSeg_Prediction')

            utils.save_volume(final_pred, affine, path)
        pass
