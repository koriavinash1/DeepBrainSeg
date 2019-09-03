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
