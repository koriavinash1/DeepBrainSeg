Windows 10 Installation Guide, by [God Bennett](https://github.com/JordanMicahBennett). 
===========

Firstly, thanks to the [author](https://github.com/koriavinash1) for the hard work on DeepBrainSeg!

Albeit, the DeepBrainSeg Library does not facilitate straight-forward Windows 10 installation. The example images [in the original repository](https://github.com/koriavinash1/DeepBrainSeg/) are Mac based windows, and [the code in the original repostory also has Mac based programming comments](https://github.com/koriavinash1/DeepBrainSeg/blob/master/ui/DeepBrainSegUI.py), like "/usr/bin/env python", which seems to indicate that Mac was used for development of this product. 

That said, here is a Windows 10 Guide, by [myself](https://github.com/JordanMicahBennett). If anyone faces any issues, let me know. ([For those curious why God Bennett is commenting from Jordan's profile, I legally changed my name from Jordan to God](https://www.researchgate.net/publication/342328687_Why_I_an_atheist_legally_changed_my_name_to_God)).


![image](https://github.com/JordanMicahBennett/DeepBrainSeg/blob/master/DeepBrainSegUI-running-on-Windows-10.gif)
Figure_0: DeepBrainSeg starting up on Windows 10.

![image](https://github.com/JordanMicahBennett/DeepBrainSeg/blob/master/Segmentation_Inference-DeepBrainSegUI-running-on-Windows-10.gif)

Figure_1: DeepBrainSeg used to get brain segmentation on Windows 10. (Note: Segmentation takes maybe roughly 1 hour via torch_cpu, on my i76700 intel cpu!!)

1. Don't run [DeepBrainSeg setup.py](https://github.com/koriavinash1/DeepBrainSeg/blob/master/setup.py) , until step 10. It matters not if you had already ran it, still proceed if you had already done so.

2. Install [Python 3.5.4 64 bit.](https://www.python.org/downloads/release/python-354/) You will notice below that everything going forward is 64 bit based!

- Failure to do the above will result in python 3.5 base version related torch error, and python 3.6 will yield DeepBrainSeg related deepSeg import error! Not to mention, anaconda cloud has no distribution of pydensecrf for python3.7, and I will mention why anaconda cloud is relevant below. Stay with python 3.5 [as advised by author](https://github.com/koriavinash1/DeepBrainSeg/issues/9#issuecomment-576507447).

3. Install pydensecrf (by simply manually copying "**Lib/site packages**" folder to "**Lib/site packages**" in Python35 installation directory), particularly "[win-64/pydensecrf-1.0rc2](https://anaconda.org/conda-forge/pydensecrf/1.0rc2/download/win-64/pydensecrf-1.0rc2-py36_0.tar.bz2)" (crucially, from [the anaconda cloud location](https://anaconda.org/conda-forge/pydensecrf/files)).

- Failure to do the above will result in pydensecrf related build errors, in conjunction with c++ dependencies, if pip is used instead of the manual process above.

4. Install pyradiomics (by simply manually copying "**Lib/site packages**" and "**Scripts**" folders to "**Lib/site packages**" and "**Scripts**" in Python35 installation directory) "[win-64/pyradiomics-2.1.0-py35_0.tar](https://anaconda.org/Radiomics/pyradiomics/2.1.0/download/win-64/pyradiomics-2.1.0-py35_0.tar.bz2)" (crucially from [the anaconda cloud location](https://anaconda.org/Radiomics/pyradiomics/files), because build errors come from attempt using pip)

- Failure to do the above will result in pydensecrf related build errors, in conjunction with c++ dependencies, if pip is used instead of the manual process above.

5. Install [visual studio build tools](https://go.microsoft.com/fwlink/?LinkId=691126). 

6. Install in particular, torch 1.2.0 from the location below, using the command below given python35 path is set:

`python -m pip install https://download.pytorch.org/whl/cpu/torch-1.2.0%2Bcpu-cp35-cp35m-win_amd64.whl`

7. Install (by simply manually copying "**Lib/site packages**" folder to "**Lib/site packages**" in Python35 installation directory), in particular, torchvision 0.4.0 , "[win-64/torchvision-0.4.0-py35_cpu](https://anaconda.org/pytorch/torchvision/0.4.0/download/win-64/torchvision-0.4.0-py35_cpu.tar.bz2)" (crucially, from [page 4 of the anaconda cloud location](https://anaconda.org/pytorch/torchvision/files?page=4)).

- Failure to do the above will result in pydensecrf related build errors, in conjunction with c++ dependencies, if pip is used instead of the manual process above.

8. It is crucial that the items in (6) and (7) are installed above, no other version, unless you verify that any other versions of torch and torchvision you install match. 

- Failure to do the above will result in import issues, including "Optional" error seen below:

![image](https://user-images.githubusercontent.com/3666405/87859067-24056780-c8f8-11ea-8c70-94e467315e79.png)

9. Test your torch, torchvision, and pydensecrf installations below:


> import torch
> 
> import torchvision
> 
> import pydensecrf
> 
> import pydensecrf.densecrf 
> #The above "import pydensecrf.densecrf " test may fail even if the one above succeeds! As advised in (8) torch and torchvision must match. If a "variable_length" issue error pops up, this means that numpy requires updating to the latest version. Typical pip works.
> 

10. Finally run [DeepBrainSeg setup.py](https://github.com/koriavinash1/DeepBrainSeg/blob/master/setup.py). Install any other missing thing if applicable, using typical python pip.

You should see DeepBrainSeg installation being resolved below:
![image](https://user-images.githubusercontent.com/3666405/87859288-a0e51100-c8f9-11ea-97f6-17b476213dec.png)


11. If no errors happen in the test, then it's time to launch the DeepBrainSeg user interface on Windows 10:

`
python DeepBrainSegUI.py #as advised on the main repository or simply run the python file from IDLE35.
`

The result should be similar to what you see in [the first](https://github.com/JordanMicahBennett/DeepBrainSeg/blob/master/DeepBrainSegUI-running-on-Windows-10.gif) and [second](https://github.com/JordanMicahBennett/DeepBrainSeg/blob/master/Segmentation_Inference-DeepBrainSegUI-running-on-Windows-10.gif) images of this readme.

*Note well/tips*: 
===========

* The trained models are automatically downloaded to **"~/.DeepBrainSeg"** (or home on mac), but automatically downloaded to **"C:\Users\<YourUserName>\.DeepBrainSeg"** on Windows!
* Ants mask generator, which needs to be used to generate a mask file for inference/real time test, is located in **"DeepBrainSeg-master/DeepBrainSeg/brainmask/antsmask.py"**
    * Crucially, follow [these guidelines](https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Windows-10) to install ANTS on Windows. Change the "/opt/ANTS/bin" directory in **"DeepBrainSeg-master/DeepBrainSeg/brainmask/antsmask.py"** to point to wherever you installed ANTS. 
* Sample non-graphical ui based segmentation sample located in **"DeepBrainSeg-master/examples/tumorsegmentation.py"**, which reads a sample from **"DeepBrainSeg-master_/sample_volume/brats"**.
