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

if config['DBS_ANTS']:
	print ("[INFO: DeepBrainSeg] (" + strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + ") " + 'Ants Installation')

	ants_path = os.path.join(home, '.DeepBrainSeg/ants')
	current_path = os.getcwd()
	if (not os.path.exists(ants_path)) or (os.listdir(ants_path) == []):
		os.makedirs(os.path.join(ants_path, 'code')) 
		os.chdir(os.path.join(ants_path, 'code'))
		os.system('git clone https://github.com/stnava/ANTs.git')
		os.makedirs(os.path.join(ants_path, 'bin/antsBuild'))
		os.chdir(os.path.join(ants_path, 'bin/antsBuild'))

		# CMake 3.15.3 installation
		print ("[INFO: DeepBrainSeg] (" + strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + ") " + 'Cmake Latest version installation')
		os.system('wget https://github.com/Kitware/CMake/releases/download/v3.15.3/cmake-3.15.3.tar.gz')
		os.system('tar xvzf cmake-3.15.3.tar.gz')
		os.chdir('cmake-3.15.3')
		os.system('./bootstrap')
		os.system('make')
		os.system('sudo make install')

		# ANTs installation
		os.system('cmake '+os.path.join(ants_path, 'code/ANTs'))
		os.system('make -j 2')

		os.chdir(os.path.join(ants_path, 'bin/antsBuild/ANTS-build')) 
		os.system('make install 2>&1 | tee install.log')

	os.chdir(current_path)


bet_path = os.path.join(home, '.DeepBrainSeg/bets')
current_path = os.getcwd()
if (not os.path.exists(bet_path)) or (os.listdir(bet_path) == []):
    os.makedirs(bet_path, exist_ok=True)
    os.chdir(bet_path)
    os.system('git clone https://github.com/MIC-DKFZ/HD-BET')
    os.chdir(os.path.join(bet_path, 'HD-BET'))
    os.system('pip install -e .')
		



