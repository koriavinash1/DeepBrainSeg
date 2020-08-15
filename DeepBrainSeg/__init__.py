from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__license__ = 'MIT'
__maintainer__ = ['Avinash Kori']
__email__ = ['koriavinash1@gmail.com']


import os
import sys
from time import gmtime, strftime
from google_drive_downloader import GoogleDriveDownloader as gdd
from os.path import expanduser
home = expanduser("~")
import json

with open(os.path.join(home, ".DeepBrainSeg/config.json", "w")) as write_file:
    config = json.load(write_file)
