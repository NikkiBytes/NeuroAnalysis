

import glob
import os
from IPython.core import display as ICD
from multiprocessing import Pool



"""
## ROI Timeseries Pull Helper Functions
Definitions of the timeseries programs common functions.
"""

############################################################################################################

def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]

############################################################################################################


def set_variables():
    # load and set variables

    # get data folder file path
    data_path = '/projects/niblab/experiments/bbx/data'

    # initialize empty variables
    data_dict = {}

    return data_dict, data_path;

############################################################################################################


def get_input_data(data_dict, data_path, fill_dict=True, verbose=True):
    # get subject functional data
    # -- assumes a certain structure, may need to modify grab for better versatility
    functional_niis = glob.glob(os.path.join(data_path, "preprocessed/sub-*/ses-1/func/*brain.nii.gz"))
    print("[INFO] {} functional nifti files found".format(len(functional_niis)))

    # fill dictionary
    if fill_dict == True:
        # data_dict['func_niis'] = functional_niis
        # loop through functionals found and fill in dictionary
        for func_file in functional_niis:
            # print(func_file.split("/")[-1])

            subj_id = func_file.split("/")[-1].split("_")[0]
            task_id = func_file.split("/")[-1].split("_")[2]
            # print(subj_id, task_id)
            if subj_id not in data_dict:
                data_dict[subj_id] = {}
            if task_id not in data_dict[subj_id]:
                data_dict[subj_id][task_id] = {}

        print("[INFO] Dictionary made, {} keys(subjects) found".format(len(data_dict.keys())))

        if verbose == True:
            print("[INFO] Dictionary keys(subjects): \n%s" % data_dict.keys())

############################################################################################################

"""
Transform niftis into correct shape using FSLs flirt command with fslmaths.
We default threshold at 0.9, with the chocolate decoding asymmetrical atlas.
"""

def tranform_niftis(niftis, verbose=False, run_process=True):

    # get the reference atlas file path
    reference_nifti = '/projects/niblab/parcellations/chocolate_decoding_rois/mni2ace.nii.gz'
    reference_mat = '/projects/niblab/parcellations/chocolate_decoding_rois/mni2ace.mat'

    # loop through functionals
    for nii in niftis:
        # setup and run flirt
        nii = nii.replace('.nii.gz', '')
        out = nii + '_3mm'

        # flirt command
        flirt_cmd = "flirt -in {} -ref {} -init {} -applyxfm -out {}".format(nii, reference_nifti, reference_mat, out)
        if verbose != False:
            print('[INFO] flirt command: \n{}'.format(flirt_cmd))
        if run_process == True:
            os.system(flirt_cmd)

        # fslmaths command to threshold --use binary option for transforming masks
        fslmaths_cmd = 'fslmaths {} -thr 0.9 {}'.format(out, out)
        if verbose != False:
            print('[INFO] fslmaths command: \n{}'.format(fslmaths_cmd))
        if run_process == True:
            os.system(fslmaths_cmd)

############################################################################################################


def pull_timeseries(file_list, bb300_path='/projects/niblab/parcellations/bigbrain300',
                    roi_df='/projects/niblab/parcellations/bigbrain300/renaming.csv'):
    bad_subs = []
    # ICD.display(roi_df)

    # load asymmetrical nifti roi files
    asym_niftis = glob.glob("/projects/niblab/parcellations/bigbrain300/MNI152Asymmetrical_3mm/*.nii.gz")

    # load roi list
    out_dir = os.path.join(data_path, 'rois/bigbrain300/funcs_uc')
    # print('[INFO] output folder: \t%s \n'%out_dir)

    # loop through the roi file list
    # print(roi_list[:3])
    for nifti in sorted(file_list):

        subj_id = nifti.split("/")[-1].split("_")[0]
        task_id = nifti.split("/")[-1].split("_")[2]
        # print('[INFO] roi: %s %s \n%s'%(subj_id, task_id, nifti))

        # loop through roi reference list
        for ref_nifti in sorted(asym_niftis):
            # print('[INFO] reference roi: %s'%ref_nifti)
            roi = ref_nifti.split('/')[-1].split(".")[0]
            out_path = os.path.join(out_dir, "{}_{}_{}_{}.txt".format(subj_id, "ses-1", task_id, roi))
            # print(roi, out_path)
            cmd = 'fslmeants -i {} -o {} -m {}'.format(nifti, out_path, ref_nifti)
            try:
                # cmd='fslmeants -i {} -o {} -m {}'.format(nifti, out_path, ref_nifti)
                print("Running shell command: {}".format(cmd))
                # os.system(cmd)
                pass
            except:
                bad_subs.append((subj_id, task_id))

        # print('[INFO] finished processing for %s'%subj_id)

    return "%s" % bad_subs

############################################################################################################

"""
# Pull Timeseries 
"""
def pull_timeseries(file_list, bb300_path='/projects/niblab/parcellations/bigbrain300',
                    roi_df='/projects/niblab/parcellations/bigbrain300/renaming.csv',
                    verbose=True, run_process=False):

    # initialize variables
    bad_subs = []

    if verbose == True:
        ICD.display(roi_df)

    # load asymmetrical nifti roi files
    asym_niftis = glob.glob("/projects/niblab/parcellations/bigbrain300/MNI152Asymmetrical_3mm/*.nii.gz")

    # load roi list
    out_dir = os.path.join(data_path, 'timeseries/bigbrain300/funcs_uc')
    if verbose == True:
        print('[INFO] output folder: \t%s \n' % out_dir)

    # loop through the roi file list
    for nifti in sorted(file_list):

        subj_id = nifti.split("/")[-1].split("_")[0]
        task_id = nifti.split("/")[-1].split("_")[2]
        # print('[INFO] roi: %s %s \n%s'%(subj_id, task_id, nifti))

        # loop through roi reference list
        for ref_nifti in sorted(asym_niftis):
            # print('[INFO] reference roi: %s'%ref_nifti)
            roi = ref_nifti.split('/')[-1].split(".")[0]
            out_path = os.path.join(out_dir, "{}_{}_{}_{}.txt".format(subj_id, "ses-1", task_id, roi))
            # print(roi, out_path)
            cmd = 'fslmeants -i {} -o {} -m {}'.format(nifti, out_path, ref_nifti)
            try:
                if verbose == True:
                    print("Running shell command: {}".format(cmd))
                if run_process == True:
                    os.system(cmd)

            except:
                bad_subs.append((subj_id, task_id))

        if verbose == True:
            print('[INFO] finished processing for %s' % subj_id)

    return "%s" % bad_subs

############################################################################################################
"""
"""

def fsl_transform(niftis, chunksize=10, poolsize=12):
    print('[INFO] transforming functional file shape to match the mask....')
    print("[INFO] breaking data into chunks, with chunksize: {}".format(chunksize))
    chunk_list = chunks(niftis, chunksize)
    print('[INFO] length of chunk list:', len(chunk_list))
     with Pool(poolsize) as p:
        p.map(tranform_niftis, chunk_list)
    print('[INFO] transformation process complete.')

############################################################################################################
"""
# Main Program
"""
# Step 1: set variables
data_dict, data_path=set_variables()

# Step 2: get functional input and setup data dictionary
get_input_data(data_dict, data_path, fill_dict=True, verbose=False)

# Step 3: transform niftis w/ FSLs flirt and fslmaths
run_transform=False
if run_transform==True:
    niftis = glob.glob(os.path.join(data_path, "preprocessed/sub-*/ses-1/func/*brain.nii.gz"))
    fsl_transform(niftis, chunksize=10, poolsize=12)

# Step 4: submit slurm batch job to pull individual bigbrain300 ROIs from subject/condition
# with FSL fslmeants command
submit_roi_job=False
if submit_roi_job==True:
    print('[INFO] submitting individual roi timeseries pull slurm job....')
    # set batch command to submit slurm job
    cmd_batch = sbatch /projects/niblab/experiments/bbx/code/batch_jobs/timeseries_pull.job
    # submit batch command
    os.system(cmd_batch)