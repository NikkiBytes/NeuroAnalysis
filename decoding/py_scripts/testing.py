import os
import numpy as np
import nilearn
import glob
import nibabel as nib
import pandas as pd 
from sklearn.model_selection import cross_val_score
from nilearn.input_data import NiftiMasker 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')

import multiprocessing
from multiprocessing import Pool
import time
from sklearn import preprocessing

#image mask
imag_mask='/projects/niblab/bids_projects/Experiments/ChocoData/images/bin_mask.nii.gz'

#our behavioral csv file 
stim = '/projects/niblab/bids_projects/Experiments/ChocoData/behavorial_data/all_waves_b1.csv'

#our dataset concatenated image 
dataset='/projects/niblab/bids_projects/Experiments/ChocoData/images/all_waves_b1.nii.gz'

#load behavioral data into a pandas df
behavioral = pd.read_csv(stim, sep="\t")


# look at original unique labels 
print(behavioral["Label"].unique())


#grab conditional labels and set up milkshake
behavioral["Label"] = behavioral.replace(['HF_LS_receipt', 'LF_LS_receipt', 'LF_HS_receipt', 'HF_HS_receipt'], 'milkshake')


y = behavioral["Label"]
print(y.unique()) # make sure all the milkshake receipts have been replaced with "milkshake"


#restrict data to our target analysis 
condition_mask = behavioral["Label"].isin(['milkshake', "h20_receipt"])
y = y[condition_mask]


#confirm we have the # of condtions needed
print(y.unique())


masker = NiftiMasker(mask_img=imag_mask, standardize=True, memory="nilearn_cache", memory_level=5)
X = masker.fit_transform(dataset)
# Apply our condition_mask
X = X[condition_mask]


# PREDICTION FUNCTION
from sklearn.svm import SVC
svc = SVC(kernel='linear', max_iter=1000)

# FEATURE SELECTION
feature_selection = SelectKBest(f_classif, k=500)


anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])


anova_svc.fit(X,y)
y_pred = anova_svc.predict(X)

##########################################################################

# NESTED CROSS VALIDATION
from sklearn.model_selection import GridSearchCV
k_range = [[ 15, 50, 150, 300], [500, 1000, 3000, 5000]]

#cv_scores = cross_val_score(anova_svc, X, conditions,)
# Print the results




def run_CV(params):
    grid = GridSearchCV(anova_svc, param_grid={'anova__k': params}, verbose=1, cv=5, n_jobs=3)
    return cross_val_score(grid, X, y, cv=5, n_jobs=3)
        
  

p = Pool(processes = 4) 
start = time.time()
nested_cv_scores =  p.map(run_CV, k_range)
p.close()
p.join()
print("Nested CV score: %.4f" % np.mean(nested_cv_scores))
print("Time: ", (time.time() - start))