import os
import sys
sys.path.append('..')
from glob import glob
from DeepBrainSeg.registration import Coregistration
from DeepBrainSeg.helpers.dcm2niftii import convertDcm2nifti
from DeepBrainSeg.brainmask.antsmask import get_ants_mask
from DeepBrainSeg.tumor import tumorSeg

coreg = Coregistration()
segmentor = tumorSeg()
dcm_subject_root = '../sample_volume/dcm/all_patients'
dcm_subjects = [os.path.join(dcm_subject_root, sub) for sub in os.listdir(dcm_subject_root)]

for subject in dcm_subjects:
    seqs = os.listdir(subject)
    json = {}
    
    for seq in seqs:
        if seq.__contains__('sT1W_FS 3D_ISO_COR'):
            json['t1c'] = os.path.join(subject, seq)
        elif seq.__contains__('T2W_FLAIR_TRA'):
            json['flair'] =  os.path.join(subject, seq)
        elif seq.__contains__('VT1W_3D_FFE'):
            json['t1'] = os.path.join(subject, seq)
        elif seq.__contains__('T2W_TSE_TRA'):
            json['t2'] =  os.path.join(subject, seq)

    
    # convert dcm to nifty
    convertDcm2nifti(path_json = json,
                    output_dir = os.path.join('../sample_results/nifty/', subject.split('/').pop()),
                    verbose = True)

    # ANTs mask extraction

    for key in json.keys():
        get_ants_mask(os.path.join('../sample_results/nifty/', subject.split('/').pop(), key+'.nii.gz'), 
			os.path.join('../sample_results/skull_strip/{}/{}.nii.gz'.format(subject.split('/').pop(), key)))

    # Coregistration
    moving_imgs = {'t1': os.path.join('../sample_results/skull_strip/{}/{}.nii.gz'.format(subject.split('/').pop(), 't1')),
                    't2': os.path.join('../sample_results/skull_strip/{}/{}.nii.gz'.format(subject.split('/').pop(), 't2')),
                    'flair':os.path.join('../sample_results/skull_strip/{}/{}.nii.gz'.format(subject.split('/').pop(), 'flair'))
                    }
    fixed_img =  os.path.join('../sample_results/skull_strip/{}/{}.nii.gz'.format(subject.split('/').pop(), 't1c'))
    coreg.register_patient(moving_images = moving_imgs,
                            fixed_image  = fixed_img,
                            save_path  = os.path.join('../sample_results/coreg/{}'.format(subject.split('/').pop())))

    # Segmentation
    # segmentor.get_segmentation(os.path.join('../sample_results/coreg/{}/isotropic/t1.nii.gz'.format(subject.split('/').pop())),
    #                             os.path.join('../sample_results/coreg/{}/isotropic/t2.nii.gz'.format(subject.split('/').pop())), 
    #                            os.path.join('../sample_results/coreg/{}/isotropic/t1c.nii.gz'.format(subject.split('/').pop())), 
    #                            os.path.join('../sample_results/coreg/{}/isotropic/flair.nii.gz'.format(subject.split('/').pop())))

