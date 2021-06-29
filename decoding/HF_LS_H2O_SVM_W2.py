
# coding: utf-8

# In[ ]:


# coding: utf-8

# In[ ]:


#this imports all the commands needed for the script to work#
import os
import numpy as np
import nilearn
import glob
#import matplotlib
import nibabel as nib
import pandas as pd 
from nilearn.input_data import NiftiMasker 
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')



#image mask
imag_mask='/projects/niblab/bids_projects/Experiments/ChocoData/images/bin_mask.nii.gz'


# In[ ]:


#our behavioral csv file 
stim = '/projects/niblab/bids_projects/Experiments/ChocoData/behavorial_data/w2_milkshake_all.csv'
#our dataset concatenated image 
dataset='/projects/niblab/bids_projects/Experiments/ChocoData/images/w2_milkshake_all.nii.gz'
#load behavioral data into a pandas df
behavioral = pd.read_csv(stim, sep="\t")
#grab conditional labels 
y = behavioral["Label"]


# In[ ]:


#restrict data to our target analysis 
condition_mask = behavioral["Label"].isin(['HF_LS_receipt', "h20_receipt"])
y = y[condition_mask]
#confirm we have the # of condtions needed
print(y.unique())

session = behavioral[condition_mask].to_records(index=False)
print(session.dtype.names)


# In[ ]:


masker = NiftiMasker(mask_img=imag_mask,standardize=True, memory="nilearn_cache", memory_level=3)
X = masker.fit_transform(dataset)
# Apply our condition_mask
X = X[condition_mask]


# In[ ]:


from sklearn.svm import SVC
svc = SVC(kernel='linear')

from sklearn.feature_selection import SelectKBest, f_classif
feature_selection = SelectKBest(f_classif, k=500)

# We have our classifier (SVC), our feature selection (SelectKBest), and now,
# we can plug them together in a *pipeline* that performs the two operations
# successively:
from sklearn.pipeline import Pipeline
anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])


anova_svc.fit(X,y)
y_pred = anova_svc.predict(X)


# In[ ]:


from sklearn.model_selection import cross_val_score

k_range = [10, 15, 30, 50, 150, 300, 500, 1000, 1500, 3000, 5000]
    
# Here we run gridsearch
from sklearn.model_selection import GridSearchCV
# We are going to tune the parameter 'k' of the step called 'anova' in
# the pipeline. Thus we need to address it as 'anova__k'.

# Note that GridSearchCV takes an n_jobs argument that can make it go
# much faster'
grid = GridSearchCV(anova_svc, param_grid={'anova__k': k_range}, verbose=1, n_jobs=1, cv=5)
nested_cv_scores = cross_val_score(grid, X, y, cv=5, groups=session)

print("Nested CV score: %.4f" % np.mean(nested_cv_scores))


# In[ ]:


# Here is the image 
coef = svc.coef_
# reverse feature selection
coef = feature_selection.inverse_transform(coef)
# reverse masking
weight_img = masker.inverse_transform(coef)


# Use the mean image as a background to avoid relying on anatomical data
from nilearn import image
mean_img = image.mean_img(dataset)
mean_img.to_filename('/projects/niblab/bids_projects/Experiments/ChocoData/derivatives/images/w2_HF_LS_milk_mean_nimask.nii')

# Create the figure
from nilearn.plotting import plot_stat_map, show
display = plot_stat_map(weight_img, mean_img, title='SVM weights HF_LS vs h2O for wave 2')
display.savefig('/projects/niblab/bids_projects/Experiments/ChocoData/derivatives/images/w2_HF_LS_milk_SVM_nimask.png')
# Saving the results as a Nifti file may also be important
weight_img.to_filename('/projects/niblab/bids_projects/Experiments/ChocoData/derivatives/images/w2_HF_LS_milk_SVM_nimask.nii')

