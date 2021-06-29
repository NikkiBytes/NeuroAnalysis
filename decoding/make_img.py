import os
import glob
import numpy as np
import nibabel as nib
from random import shuffle
import pandas as pd


wave4 = sorted(glob.glob(os.path.join("/projects/niblab/bids_projects/Experiments/ChocoData/derivatives/sub-*/ses-4/func/Ana*/feat1/task-*/filtered_func_data.nii.gz")))
wave3 = sorted(glob.glob(os.path.join("/projects/niblab/bids_projects/Experiments/ChocoData/derivatives/sub-*/ses-3/func/Ana*/feat1/task-*/filtered_func_data.nii.gz")))
wave2 = sorted(glob.glob(os.path.join("/projects/niblab/bids_projects/Experiments/ChocoData/derivatives/sub-*/ses-2/func/Ana*/feat1/task-*/filtered_func_data.nii.gz")))
wave1 = sorted(glob.glob(os.path.join("/projects/niblab/bids_projects/Experiments/ChocoData/derivatives/sub-*/ses-1/func/Ana*/feat1/task-*/filtered_func_data.nii.gz")))

subj4 = [i.split("/")[-7] for i in wave4]
subj3 = [i.split("/")[-7] for i in wave3]
subj2 = [i.split("/")[-7] for i in wave2] 
subj1 = [i.split("/")[-7] for i in wave1]

# remove duplicates 
subj4 = list(set(subj4))
subj3 = list(set(subj3))
subj2 = list(set(subj2))
subj1 = list(set(subj1))



global outpath
global DERIV_DIR
outpath = "/projects/niblab/bids_projects/Experiments/ChocoData/images"
DERIV_DIR = '/projects/niblab/bids_projects/Experiments/ChocoData/derivatives'


# SETTING UP IMAGE WITH ALLL 4 WAVES
# step one is to get all subject available in all waves
sub_list = [i for i in subj4 if i in (subj3 and subj2 and subj1)]
sub_list = sorted(sub_list)

filename = "exc_full_set.nii.gz"
all_funcs = []
for sub in sub_list:
    sub_funcs = glob.glob(os.path.join(DERIV_DIR, sub, "ses-*/func/Analysis/feat1/task*/filtered_func_data.nii.gz"))
    for y in sub_funcs:
        all_funcs.append(y)
all_funcs = sorted(all_funcs)
ni2_funcs = (nib.Nifti2Image.from_image(nib.load(func)) for func in all_funcs)
ni2_concat = nib.concat_images(ni2_funcs, check_affines=False, axis=3)
outfile=os.path.join(outpath, filename)
#write the file
ni2_concat.to_filename(outfile)


##############################################
######### Set up the Behavioral CSV ##########
# Step one -get subjects

subj_dict = {}
for x in sub_list: 
    subj_dict[x] = [] 
    
# -add corresponding milkshake label    
ct=0
for d in subj_dict:
    task_dirs = glob.glob(os.path.join('/projects/niblab/bids_projects/Experiments/ChocoData/derivatives', d, "ses-*/func/Analysis/feat1/*"))
    for task in task_dirs:
        dir_name = task.split("/")[-1]
        sess = task.split("/")[-5]
        mlk = dir_name.split(".")[0].split("e")[1]
        id_ = sess+"_"+mlk
        subj_dict[d].append(id_)
        ct=ct+1
        
temp_out = "/projects/niblab/bids_projects/Experiments/ChocoData/behavorial_data/temp"
for sub in subj_dict:
    file = open(temp_out+"/tempfile_%s.txt"%sub, 'a')
    for label in sorted(subj_dict[sub]):
        print(label.split("_")[1])
        fileX = '/projects/niblab/data/eric_data/ev_files/milkshake/mk%s_attr.txt'%label.split("_")[1]
        fileX_contents = open(fileX, 'r')
        data = fileX_contents.read()
        fileX_contents.close()
        file.write(data)
    file.close()
    
tempfiles=sorted(glob.glob("/projects/niblab/bids_projects/Experiments/ChocoData/behavorial_data/temp/tempfile_*.txt"))

# fill in 20% of test files 
test_slice_count = round(len(tempfiles) *.2)
test_slice = sub_list[:test_slice_count]
for sub in test_slice:
    df=pd.read_csv("/projects/niblab/bids_projects/Experiments/ChocoData/behavorial_data/temp/tempfile_%s.txt"%sub, sep=' ',header=None)
    df[1].replace(0,1, inplace=True)
    print(df.head())
    df.to_csv("/projects/niblab/bids_projects/Experiments/ChocoData/behavorial_data/temp/tempfile_%s.txt"%sub, sep=' ', index=False, header=None)    

# Make initial text file    
fileout = open("/projects/niblab/bids_projects/Experiments/ChocoData/behavorial_data/exc_full_set.txt", "a")
fileout.write("Label sess\n")
       
for f in tempfiles:
    print("adding file %s"%f)
    f_contents = open(f, "r")
    data=f_contents.read()
    f_contents.close()
    fileout.write(data)
    
fileout.close()
# Make final CSV 
# MAKE CSV FILE WITH PANDAS
import pandas as pd
df=pd.read_csv("/projects/niblab/bids_projects/Experiments/ChocoData/behavorial_data/exc_full_set.txt", sep=' ')
df.to_csv("/projects/niblab/bids_projects/Experiments/ChocoData/behavorial_data/exc_full_set.csv", sep='\t', index=False)


