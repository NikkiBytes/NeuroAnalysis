import os
import numpy as np
import nilearn
import glob
import nibabel as nib
import pandas as pd 
from sklearn.model_selection import cross_val_score
from nilearn.input_data import NiftiMasker 
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore', 'Solver terminated early.*')
from sklearn.model_selection import GridSearchCV
import time

#image mask
imag_mask='/projects/niblab/bids_projects/Experiments/ChocoData/images/bin_mask.nii.gz'

#our behavioral csv file 
stim = '/projects/niblab/bids_projects/Experiments/ChocoData/behavorial_data/milkshake_all.csv'

#our dataset concatenated image 
dataset='/projects/niblab/bids_projects/Experiments/ChocoData/images/milkshake_all.nii.gz '

#load behavioral data into a pandas df
behavioral = pd.read_csv(stim, sep="\t")



#grab conditional labels and set up milkshake

behavioral["Label"] = behavioral.replace(['HF_LS_receipt', 'LF_HS_receipt', 'HF_HS_receipt'], 'milkshake')

y = behavioral["Label"]
print(y.unique())


#restrict data to our target analysis 
condition_mask = behavioral["Label"].isin(['milkshake', "h20_receipt"])
y = y[condition_mask]
#confirm we have the # of condtions needed
print(y.unique())


masker = NiftiMasker(mask_img=imag_mask, standardize=True, memory="nilearn_cache", memory_level=3)
X = masker.fit_transform(dataset)
# Apply our condition_mask
X = X[condition_mask]

scaler = StandardScaler()
X = scaler.fit_transform(X)



from sklearn.svm import SVC
svc = SVC(kernel='linear', max_iter=1500)

from sklearn.feature_selection import SelectKBest, f_classif
feature_selection = SelectKBest(f_classif, k=500)

# We have our classifier (SVC), our feature selection (SelectKBest), and now,
# we can plug them together in a *pipeline* that performs the two operations
# successively:
from sklearn.pipeline import Pipeline


anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])
anova_svc.fit(X,y)
y_pred = anova_svc.predict(X)

start = time.time()
print("running nested CV......")
k_range = [ 15, 50, 150, 500, 1000, 3000, 5000]
grid = GridSearchCV(anova_svc, param_grid={'anova__k': k_range}, verbose=1, cv=5, n_jobs=4)
nested_cv_scores = cross_val_score(grid, X, y, cv=5, n_jobs=4)
#NEST_SCORE = np.mean(nested_cv_scores)
print("Nested CV score: %.4f" % np.mean(nested_cv_scores))



# Here is the image 
coef = svc.coef_
# reverse feature selection
coef = feature_selection.inverse_transform(coef)
# reverse masking
weight_img = masker.inverse_transform(coef)


# Use the mean image as a background to avoid relying on anatomical data
from nilearn import image
mean_img = image.mean_img(dataset)
mean_img.to_filename('/projects/niblab/bids_projects/Experiments/ChocoData/derivatives/code/decoding/milkshake_vs_h2O/images/4w_p2_mean_nimask.nii')

# Create the figure
from nilearn.plotting import plot_stat_map, show
display = plot_stat_map(weight_img, mean_img, title='Milkshake vs. H2O')
display.savefig('/projects/niblab/bids_projects/Experiments/ChocoData/derivatives/code/decoding/milkshake_vs_h2O/images/4w_p2_SVM_nimask.png')
# Saving the results as a Nifti file may also be important
weight_img.to_filename('/projects/niblab/bids_projects/Experiments/ChocoData/derivatives/code/decoding/milkshake_vs_h2O/images/4w_p2_SVM_nimask.nii')
print("Time: ", (time.time() - start))