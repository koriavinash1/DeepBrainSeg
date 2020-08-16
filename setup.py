import setuptools
import os, json
from os.path import expanduser
home = expanduser("~")

config = {
    'DBS_ANTS': False,
}

os.makedirs(os.path.join(home, ".DeepBrainSeg"), exist_ok=True)
with open(os.path.join(home, ".DeepBrainSeg/config.json"), "w") as write_file:
    json.dump(config, write_file)


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='DeepBrainSeg',  
     version='0.2.0',
     author="Avinash Kori",
     author_email="koriavinash1@gmail.com",
     description="Deep Learning tool for brain tumor segmentation.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/koriavinash1/DeepBrainSeg",
     packages=setuptools.find_packages(),
     install_requires = [
         'torch==1.5.1',
         'torchvision==0.6.1',
         'torchnet==0.0.4',
         'nibabel==3.0.2',
         'SimpleITK==1.2.4',
         'tqdm==4.48.2',
         'numpy==1.18.5',
         'pandas==0.25.3',
         'scipy==1.4.1',
         'pydensecrf==1.0rc3',
         'pyradiomics==3.0',
         'scikit-image==0.15',
    	 'dicom2nifti==2.2.10',
         'googledrivedownloader==0.4',
         ],
     classifiers=[
         "Programming Language :: Python :: 3.5",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
