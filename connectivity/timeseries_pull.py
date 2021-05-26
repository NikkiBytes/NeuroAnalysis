import glob
import os
import sys
from multiprocessing import Pool
from IPython.core import display as ICD


### Helper Functions

def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def pull_timeseries(file_list, bb300_path='/projects/niblab/parcellations/bigbrain300',
                    roi_df='/projects/niblab/parcellations/bigbrain300/renaming.csv',
                    verbose=True, run_process=False):

    # initialize variables
    bad_subs=[]

    if verbose==True:
        ICD.display(roi_df)

    # load asymmetrical nifti reference ROI file
    asym_niftis=glob.glob("/projects/niblab/parcellations/bigbrain300/MNI152Asymmetrical_3mm/*.nii.gz")

    # set out path directory folder
    out_dir = os.path.join(data_path, 'timeseries/bigbrain300/funcs_uc')
    if verbose==True:
        print('[INFO] output folder: \t%s \n'%out_dir)


    # loop through the roi file list
    for nifti in sorted(file_list):

        subj_id = nifti.split("/")[-1].split("_")[0]
        task_id = nifti.split("/")[-1].split("_")[2]

        if verbose==True:
            print('[INFO] roi: %s %s \n%s'%(subj_id, task_id, nifti))

        # loop through roi reference list
        for ref_nifti in sorted(asym_niftis):
            #print('[INFO] reference roi: %s'%ref_nifti)
            roi = ref_nifti.split('/')[-1].split(".")[0]
            out_path = os.path.join(out_dir, "{}_{}_{}_{}.txt".format(subj_id, "ses-1", task_id, roi))
            #print(roi, out_path)
            cmd='fslmeants -i {} -o {} -m {}'.format(nifti, out_path, ref_nifti)
            try:
                if verbose==True:
                    print("Running shell command: {}".format(cmd))
                if run_process==True:
                    os.system(cmd)

            except:
                bad_subs.append((subj_id, task_id))

        if verbose==True:
            print('[INFO] finished processing for %s'%subj_id)


    return "%s"%bad_subs


"""  
# Timeseries Pull Main Program
"""

### Set paths and other variables

# set variables
data_path = sys.argv[1]
# chunksize_input=sys.argv[1]
# pool_size_input=sys.argv[2]

# initialize data variables
data_dict = {}
bad_subs = []


# load roi
#print("[INFO] loading roi and reference file....")
# get functionals
funcs_3mm = glob.glob(os.path.join(data_path, "preprocessed/sub-*/ses-1/func/*brain_3mm.nii.gz"))
print("[INFO] {} functional nifti files found".format(len(funcs_3mm)))
chunksize = 16
print("[INFO] chunksize: {}".format(chunksize))
chunk_list = chunks(funcs_3mm, chunksize)


# roi_df['network']
# pull timeseries by rois --fslmeants command
# print(chunk_list)
def run_process(pool_size):
    print("[INFO] starting multiprocess...")
    with Pool(pool_size) as p:
        error_subjects = p.map(pull_timeseries, chunk_list)
    print("[INFO] process complete. \n[INFO] bad subjects: \t\t%s" % error_subjects)


pool_size = 16
run_process(pool_size)


