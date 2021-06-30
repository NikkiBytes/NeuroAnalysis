"""
# Timeseries Pull with FSL Package

"""

## imports

import os, glob
import pandas as pd
import nipype.interfaces.fsl as fsl
import multiprocessing as mp
import time



class FMRITimeseries:
    """
    ## ROI Timeseries Pull Helper Functions
    Definitions of the timeseries programs common functions.
    """
    def __init__(self, func_folders):
        self.func_folders = func_folders
        self.data_dict = {}
        

    # Helper Functions

    
    ## build_chunklist() 
    def build_chunklist(self, subject_ids=[]):
        # set dataset list l
        if not subject_ids:
            l=self.func_folders
        else:
            l=subject_ids
        
        
        # break dataset into chunksize n
        if len(l) > 200:
            n=len(l)//100 # set chunksize
        elif len(l) > 50:
            n=len(l)//20 # set chunksize
        elif len(l) > 25:
            n=len(l)//10# set chunksize
        else:
            n=len(l)//5 # set chunksize
        print('[INFO] CHUNK SIZE: %s'%n)

        # grab concatenated (fslmerge) data
        chunk_list=[l[i:i+n] for i in range(0, len(l), n)]
        print('[INFO] CHUNK LIST SIZE: %s'%(len(chunk_list)))
        
        return chunk_list;

    ## make the dictionary with subject ids and task (resting or functional expected)
    def setup_dictionary(self, filtered_func=False):
        # setup a dictionary
        data_dict={}
        print("[INFO] building dictionary....")
        for folder in self.func_folders:
            #print(func_file.split("/")[-1])
            subj_id = folder.split("/")[-2]
            task_id = folder.split("/")[-1]
            
            # task string edits
            if ".feat" in task_id:
                task_id=task_id.replace('.feat', '')
            print('[INFO] subject: %s \t task: %s'%(subj_id, task_id))
            if subj_id not in data_dict:
                data_dict[subj_id]={}
            if task_id not in data_dict[subj_id]:
                data_dict[subj_id][task_id]={}
                
            if filtered_func == True:
                flt_func_file=os.path.join(folder, 'filtered_func_data.nii.gz')
                data_dict[subj_id][task_id]['FILTER_FUNC']=flt_func_file
                
        self.data_dict=data_dict
        return data_dict;

        

    ## transform nifits -- with FSL flirt
    def fsl_flirt(self,subject,target_task, verbose=False):
        # set references file paths
        reference_nifti= '/projects/niblab/parcellations/chocolate_decoding_rois/mni2ace.nii.gz'
        reference_matrix= '/projects/niblab/parcellations/chocolate_decoding_rois/mni2ace.mat'
        out_folder=os.path.join('/projects/niblab/experiments/project_milkshake/data/timeseries/folder1')
        # loop through tasks and select given task
        for subject_task in data_dict[subject]:

            # get target task
            if subject_task in target_task:
                print('[INFO] Applying a transformation to {} {} file with FSL flirt.'.format(subject,target_task))

                img=data_dict[subject][target_task]['FILTER_FUNC']
                #func_imgs = os.path.join(os.path.join())
                #print('[INFO] func img found: %s'%func_img)
                
                ## setup command input variables
                #filename=
                outfile='{}_{}_3mm.nii.gz'.format(subject, target_task)
                outfile=os.path.join(out_folder,outfile)
                #print('[INFO] out file: {}'.format(out_file)) 

                # intiialize objects
                applyxfm = fsl.preprocess.ApplyXFM()

                # set command variables
                applyxfm.inputs.in_file=img # input img filename
                applyxfm.inputs.reference=reference_nifti #input reference img filename
                applyxfm.inputs.in_matrix_file=reference_matrix # input reference matrix
                applyxfm.inputs.out_file=outfile# output img filename
                # apply transformation supplied by the matrix file
                applyxfm.inputs.apply_xfm = True 
                if verbose==True: print('[INFO] IMAGE %s FILE: \n%s'%(out_folder, applyxfm.inputs)) # check command
                ## run FSL Flirt command
                result = applyxfm.run()
                
               
                
            else:
                #print('[INFO] ERROR: target task {} not found in subject {}'.format(target_task, subject))
                pass
        
            


# *Main Program*

## Step 1: 

# set data variables

# path that holds the subjects
data_path='/projects/niblab/experiments/project_milkshake/derivatives'

# get subject task functional images (niftis)
# -- note: this code grabs functional images,  usually resting or functional-tasks. 
# This code may be unique, may be more efficient way to grab user code.
func_folders=glob.glob(os.path.join(data_path,'sub-*/*'))
# sort images
func_folders.sort()
#print(func_imgs)


# initialize fmri timeseries class object --inherites functional image list
obj1 = FMRITimeseries(func_folders)
# --test class
#print(obj1.data_dict)

# setup data dictionary with unique subject and task keys
data_dict=obj1.setup_dictionary(filtered_func=True)
#print(data_dict)                                                    


# set a list of subject ids from the dictionary 1st level keys 
subject_ids=list(data_dict.keys())
subject_ids.sort()
subject_ct=len(subject_ids) # get count of subject dataset
print("[INFO] Dictionary made, {} keys \n[INFO] Keys: {}".format(len(data_dict.keys()), subject_ids))


# build chunklist
chunk_list=obj1.build_chunklist(subject_ids=subject_ids)
#print('[INFO] CPU ct: %s'%mp.cpu_count())





multi_task=True
single_task=False

# loops through subject ids and performs a task
def subject_loop(subject):
    #subject_ids=chunklist
    #print(subject_ids)
    #for subject in subject_ids:
    
    if single_task == True:
        task='mk1'
        obj1.fsl_flirt(subject, task)
        #print(task)
    if multi_task == True:
        task_list=['mk1', 'mk2', 'mk3', 'mk4', 'mk5']

        for task in task_list:
            task=task
            #print(subject, task)
            obj1.fsl_flirt(subject, task)

    



## Step 2: Transform Functionals to match the atlas masks. 

print('[INFO] transforming functionals to match the mask with flirt....')

flirt_start_time = time.time()
for chunk in chunk_list:
    with mp.Pool(12) as p:
        #print(chunk)
        p.map(subject_loop, chunk)
print('[INFO] transformation process complete.')
flirt_process_time=time.time() - flirt_start_time
print("--- %s seconds ---"%flirt_process_time )

