from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



__license__ = 'MIT'
__maintainer__ = ['Avinash Kori']
__email__ = ['koriavinash1@gmail.com']


import os
import sys
from time import gmtime, strftime
import wget

def maybe_download(path):
    name = path.split('/')[-1]
    if not os.path.exists(path):
        print ("[INFO: DeepBrainSeg] (" + strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + ") " + name)
        wget.download('https://github.com/koriavinash1/DeepBrainSeg/releases/download/0.0.1/{}'.format(name),
            out = path)



from .Tester import *
from .finetuning import *