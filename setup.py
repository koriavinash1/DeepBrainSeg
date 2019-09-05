import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='DeepBrainSeg',  
     version='0.1.3',
     author="Avinash Kori",
     author_email="koriavinash1@gmail.com",
     description="Deep Learning tool for brain tumor segmentation.",
     long_description=open("README.md").read(),
   long_description_content_type="text/markdown",
     url="https://github.com/koriavinash1/DeepBrainSeg",
     packages=setuptools.find_packages(),
     install_requires = [
         'googledrivedownloader',
         'torch',
         'torchvision',
         'nibabel',
         'SimpleITK',
         'tqdm',
         'pandas',
         'scipy',
         'pydensecrf',
         'scikit-image'
         ],
     classifiers=[
         "Programming Language :: Python :: 3.5",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
