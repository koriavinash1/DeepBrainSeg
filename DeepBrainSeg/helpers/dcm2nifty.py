import dicom
import SimpleITK as sitk
import os
import glob
import dicom2nifti
import csv
import tqdm
col_name = ['Patient Name', 'Patient Age', 'Patient Sex',  'Day', 'Series',
            'Maker', 'Model', 'Acquisition type', 'Field Strength', 
            'Slice Thickness', 'Pixel Spacing 1', 'Pixel Spacing 2', 'Rows',
            'Columns', 'Number of Slices']
    
def find_presurgery(patient_dir):
    '''
    Function that inputs the location of a specific patient's
    directory and outputs the earliest scan folder
    
    input: patient_dir 
    
    output: presurgery_file - a string containing the location of the earliest 
    scan
    
    '''
    scan_temporal = glob.glob(patient_dir+'/*')
    flag = 1    
    for s in scan_temporal:
        seq_dir =  glob.glob(s+'/*')[0]
        dicom_filename = seq_dir+'/000000.dcm' 
        ds = dicom.read_file(dicom_filename)
        #print(ds.StudyDescription)
        t = int(ds.StudyDescription.split('D')[-1])
        if flag:
            flag = 0
            min_t = t
            presurgery_file = s
        if t<min_t:
            min_t = t
            presurgery_file = s
    #print(min_t)
    return [presurgery_file, str(ds.PatientName), min_t]

def get_dval(series_dir):
    '''
    Function to evaluate D value given a patient time series directory
    
    input: series_dir - directory with the scans at day 'D'
    
    output: The day number
    '''
    seq_dir =  glob.glob(series_dir+'/*')[0]
    dicom_filename = seq_dir+'/000000.dcm' 
    ds = dicom.read_file(dicom_filename)
    #print(ds.StudyDescription)
    return ds.StudyDescription.split()[-1]
    

def find_req_folder(data_dir):
    '''
    Function that walks through the data_dir and visits the patients folder 
    one by one and returns a list of the earliest scan directory of all 
    patients
    
    input: data_dir - directoy containing all the patients eg: 'F:\\ivy\\DOI'
        
    output: scan_dir_list - list of directories, patient name and lowest D-val
    '''
    patient_list = glob.glob(data_dir+"/*")
    scan_dir_list=[]
    for p in patient_list:
        scan_dir_list.append(find_presurgery(p))
    #print(find_presurgery(p))
    return scan_dir_list

def convert_dicom_to_nifti(data_dir, output_dir): 
    '''
    Function that converts a time series folder to the respective nifti format
    using the dicom2nifti package
    
    input: data_dir - Series directory
           ouput_dir - The destination folder
    
    '''  
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    scan_dir_list = find_req_folder(data_dir)
    
    for scan_dir,pat_name,_ in scan_dir_list:
        dest = output_dir + '/' + pat_name
        if not os.path.exists(dest):
            os.mkdir(dest)
        dicom2nifti.convert_dir.convert_directory(scan_dir, dest)
        
def get_meta_data(ds):
    '''
    Function that inputs the dicom type variable and parses the necessary meta 
    data required
    
    input: ds - The object obtained after dicom.read_file
    
    output: data - List with the required meta data
    '''
    data = []
    data.append(str(ds.PatientName))
    try:
        # TODO: Genralize for missing any attribute
        data.append(ds.PatientAge)
    except:
        data.append(-1)
    data.append(ds.PatientSex)
    data.append(int(ds.ClinicalTrialTimePointID))
    data.append(ds.SeriesDescription)
    data.append(ds.Manufacturer)
    data.append(ds.ManufacturerModelName)
    data.append(ds.MRAcquisitionType)
    data.append(float(ds.MagneticFieldStrength))
    data.append(str(ds.SliceThickness))
    data.extend(list(map(str, ds.PixelSpacing)))
       
    return data
        
def dicom_to_nifti(seq_dir, out_dir, error_file=[], num=0, csv_write=False):
    '''
    Function that converts a directory with dicom files to the nifti format 
    using sitk package. Optionally, extracts meta data and writes it to a csv 
    file
    
    input: seq_dir - Directory with the dicom files for a sequence
           out_dir - Destination directory
           error_file - List to which any file that is not executed is added to
           num - A number that will be used in the file name
           csv_write - File name of the csv file to be edited
    '''

    reader = sitk.ImageSeriesReader()    
    try:
        dicom_name = reader.GetGDCMSeriesFileNames(seq_dir)
        reader.SetFileNames(dicom_name)
        image = reader.Execute()
        dicom_filename = seq_dir+'/000000.dcm' 
        ds = dicom.read_file(dicom_filename) 
       	desc = (ds.SeriesDescription).split('(')[0]
        desc = desc.strip()

        # print (desc)
        desc = desc.replace(" ", "_")
        desc = desc.replace("/", "_")        
        
        # print (desc)
        dest = out_dir + '/'+ desc+ '_' + str(num)+'.nii'+'.gz'
        sitk.WriteImage(image, dest)
        if csv_write:            
            size = list(image.GetSize())
            data = get_meta_data(ds)
            data = data + size
            
            if not os.path.exists(csv_write):  
            	#print(10101) 

            	# Windows or python3 (one of these needs the below command)     
            	# with open(CSV_FILE, "a",newline='') as fp:

                with open(csv_write, "a") as fp:
                    wr = csv.writer(fp)
                    wr.writerow(col_name)

            # Wondows/python3
            # with open(CSV_FILE, "a",newline='') as fp:
            with open(csv_write, "a") as fp:
                wr = csv.writer(fp)
                wr.writerow(data)
    except Exception as e:
    	error_file.append(seq_dir)


def process_patient_series(series_dir, out_dir, error_list, csv_f = CSV_FILE):
    D = get_dval(series_dir)
    dest = out_dir + '/' + D
    if not os.path.exists(dest):
        os.mkdir(dest)
    seq_list = glob.glob(series_dir+'/*')   
    i = 0
    for l in seq_list:     
        i = i+1
        dicom_to_nifti(l,dest,error_list,i, csv_f)
        

def process_patient(pat_dir, output_dir, error_list):
    pat_name = os.path.split(pat_dir)[-1]
    dest = output_dir + '/' + pat_name
    if not os.path.exists(dest):
        os.mkdir(dest)
    series_list = glob.glob(pat_dir + '/*')
    for s in tqdm.tqdm(series_list):
    	
        process_patient_series(s, dest, error_list)


def convert_dir(data_dir, output_dir):
    '''
    '''
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    error_list = []
    pat_file_list  = glob.glob(data_dir+'/*')
    for s in tqdm.tqdm(pat_file_list):
        process_patient(s, output_dir, error_list)
    return error_list


if __name__=="__main__":
    convert_dir(DATA_DIR, DEST_DIR)    
    
    
        
    
    

