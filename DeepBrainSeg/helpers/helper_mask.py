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

import argparse
import os
parser = argparse.ArgumentParser(description='This piece code is used to  create Brain Mask. (a) This code requires ants to be installed. (b) The program expects flair.mha,t2.mha t1c.mha and t1c.mha within the same folder. See 1.jpg for the required folder structure')
parser.add_argument('--ants_path', type=str, default='antsbin/bin/',
                   help='path where the ants is installed, typically antsbin/bin/')
parser.add_argument('--input_path', type=str ,default='./data/Test',
                   help='location of the data')

args = parser.parse_args()

print ('The specified antspath is:',args.ants_path)
print ('The specified inputpath is',args.input_path)

input_path= args.input_path
ANTSPATH  = args.ants_path

input_path,patients, files = next(os.walk(input_path))

for p in patients:
    print ('patient name:',p)
    print
    print
    imgs = os.listdir(input_path+'/'+p)

    for i in imgs:
        input_file= input_path+'/'+p+'/'+i
        mask_file = input_path+'/'+p+'/mask.nii.gz'

        if ('t2' in i):
            os.system(ANTSPATH+'ImageMath 3 '+mask_file+' Normalize '+input_file)
            os.system(ANTSPATH+'ThresholdImage 3 '+mask_file+' '+mask_file+' 0.01 1')
            os.system(ANTSPATH+'ImageMath 3 '+mask_file+' MD '+mask_file+' 1')
            os.system(ANTSPATH+'ImageMath 3 '+mask_file+' ME '+mask_file+' 1')
            os.system(ANTSPATH+'CopyImageHeaderInformation '+input_file+' '+mask_file+' '+mask_file+' 1 1 1')

print ('mask generation completed. Please check the input directory to see the mask')
