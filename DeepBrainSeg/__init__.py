from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__license__ = 'MIT'
__maintainer__ = ['Avinash Kori']
__email__ = ['koriavinash1@gmail.com']


import os
import sys
from time import gmtime, strftime
from os.path import expanduser
home = expanduser("~")
import json


with open(os.path.join(home, ".DeepBrainSeg/config.json"), "r") as write_file:
    config = json.load(write_file)

if config['DBS_ANTS']:
	ants_path = os.path.join('/opt/ANTs/bin/')
else:
	ants_path = None