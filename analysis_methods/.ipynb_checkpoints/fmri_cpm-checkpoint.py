"""#
# Arguments : (input_path, )
# Take behavioral file, and list of functionals
# G
"""

import pandas as pd
import glob, os
import time
from IPython.display import display


class fMRIcpm:
    def __init__(self, behavioral_file, data_dict={}):
        #self.input_path=input_path
        self.behavioral_file=behavioral_file
        self.data_dict = {}
    def behavioral_prep(self, subj_key="ID", set_index=False,concat_sub=False,
                       rename_index=False, drop_subjects=False):
        
        data_dict=self.data_dict
        behavioral=pd.read_csv(self.behavioral_file)
        print("[INFO] Loaded behavioral file: {}".format(self.behavioral_file))


        ## NOTE: we have to modify the subject id here
        ## -- may be unique cases -- ##
        
        ## data transformations
                    
        if concat_sub == True:
            for id_num in behavioral[subj_key]:
                new_value='sub-%03d'%id_num
                behavioral.iloc[id_num-1, behavioral.columns.get_loc('ID')]=new_value

                
            #behavioral[subj_key] = 'sub-0'+ behavioral[subj_key].astype(str)
        if set_index == True:
            behavioral.set_index(subj_key, inplace=True)
        if rename_index == True: 
            # rename index
            behavioral.index.name="subject"

        if drop_subjects==True:    
            # gathering the subjects to drop
            drop_list=[x for x in behavioral.index.values if x not in self.subject_ids] 
        return behavioral, data_dict;
    
    
    def func_input_prep(self, func_data_path, set_dict=False):
        fcms_list = glob.glob(os.path.join(func_data_path,"*.txt"))
        subject_ids=[z.split('/')[-1].split("_")[0] for z in fcms_list]
        print('[INFO] %s subject matrices found.'%(len(subject_ids)))
        data_dict=self.data_dict
        self.subjet_ids=subject_ids
        if set_dict == True:
            for sub_id in subject_ids:
                if sub_id not in data_dict:
                    data_dict[sub_id] = {}



        for file in fcms_list:
            subj_id=file.split('/')[-1].split("_")[0]
            if subj_id not in data_dict:
                data_dict[subj_id]={
                    "FCM": file
                }

    def

        return data_dict;
        
        

