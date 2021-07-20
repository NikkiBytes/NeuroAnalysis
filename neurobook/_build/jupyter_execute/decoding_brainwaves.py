#!/usr/bin/env python
# coding: utf-8

# # Decoding Cocaine Brainwave Connectivity   
# do these plots by region with pre and post cocaine (like NAcc on top pre and NAcc on cocaine on bottom)?  Just for Beta and theta  but for all four regions on sperate graphs?

# Currently: 2 subjects (mice), 2 recordings (1-2), 2 tasks (pre-cocaine, post-cocaine).   
# Time per recording: 0-1799 seconds

# In[13]:


print("[INFO] mounting google drive directory...\n[INFO] if prompted follow the link and copy and paste the token.")
from google.colab import drive
drive.mount('/content/drive')


# In[14]:


print("[INFO] installing nilearn for session...")
get_ipython().system('pip install nilearn')


# In[ ]:


# import packages 
from sklearn.metrics import classification_report, confusion_matrix

from IPython.core import display as ICD
import pandas as pd
import glob, os
import seaborn as sns
import nilearn
from sklearn import svm
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# setting color dictionaries for maps

post_color_dict= {'PFC gamma': 'purple', 'VTA gamma':'violet', 'BLA gamma':'plum', 'NAc gamma':'mediumorchid',
             'PFC beta': 'dodgerblue', 'VTA beta':'lightskyblue', 'BLA beta':'turquoise', 
             'NAc beta':'deepskyblue', 'PFC theta': 'limegreen', 'VTA theta':'palegreen',
             'BLA theta': 'yellowgreen', 'NAc theta':'olive'}

pre_color_dict= {'PFC gamma': 'dimgray', 'VTA gamma':'gray', 'BLA gamma':'silver', 'NAc gamma':'lightgray',
             'PFC beta': 'red', 'VTA beta':'darkred', 'BLA beta':'indianred', 
             'NAc beta':'lightcoral', 'PFC theta': 'gold', 'VTA theta':'goldenrod',
             'BLA theta': 'orange', 'NAc theta':'yellow'}



# ## Data Preparation 

# In[17]:


# load the data file into a data frame 

print('[INFO] loading the multisite brainwave connectivity data into a dataframe now...')
multisite_df = pd.read_csv('drive/My Drive/Projects/pilot_mouse_connectivity/cocaine_two_mice/Spectrogram_data/updated_data/multisite_averaged_data.csv')

multisite_df.head()


# In[18]:


# drop unneccesary first column
print('INFO] cleaning data...')

multisite_df.drop("Unnamed: 0", axis=1,inplace=True)
print(multisite_df.columns.values)
multisite_df.head()


# **Factorize categorical data**  
# 
# Here we create a new column "cocaine status num", that holds 0 if the true label is post cocaine, and 1 if the true label is pre cocaine. 

# In[19]:


print('[INFO] factorizing target variable: cocaine status ...')
# turn cocaine category labels into numerics, 0: post and 1: pre
multisite_df['cocaine status num'] = pd.factorize(multisite_df['cocaine status'])[0]


# **Fill in NaN values with average**

# In[20]:


print('[INFO] filling missing data with average ...')
multisite_df.fillna(multisite_df.mean(), inplace=True)
print('[INFO] view data ...')
multisite_df.tail()


# **Normalize data**  
# Here the brainwave data is being scaled. 

# In[21]:


print('[INFO] scaling brainwave data down ...')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
multisite_df[['PFC gamma', 'VTA gamma', 'BLA gamma', 'NAc gamma',
       'PFC beta', 'VTA beta', 'BLA beta', 'NAc beta', 'PFC theta',
       'VTA theta', 'BLA theta', 'NAc theta', 'reference wires']] = scaler.fit_transform(multisite_df[['PFC gamma', 'VTA gamma', 'BLA gamma', 'NAc gamma',
                                                                                                       'PFC beta', 'VTA beta', 'BLA beta', 'NAc beta', 'PFC theta',
                                                                                                       'VTA theta', 'BLA theta', 'NAc theta', 'reference wires']])
       
print('[INFO] Scaled data:')
multisite_df.tail()


# Create a dataframe that holds the regions averaged across waves

# In[ ]:


region_avg_df=pd.DataFrame()

region_avg_df['NAc'] =  multisite_df.filter(regex=("NAc*")).mean(axis=1)
region_avg_df['PFC'] =  multisite_df.filter(regex=("PFC*")).mean(axis=1)
region_avg_df['VTA'] =  multisite_df.filter(regex=("VTA*")).mean(axis=1)
region_avg_df['BLA'] =  multisite_df.filter(regex=("BLA*")).mean(axis=1)
region_avg_df['reference wires'] = multisite_df['reference wires']
region_avg_df['mouse id'] = multisite_df['mouse id']
region_avg_df['Time (s)'] = multisite_df['Time (s)']
region_avg_df['Speed (cm/s)'] = multisite_df['Speed (cm/s)']
region_avg_df['cocaine status'] = multisite_df['cocaine status']
region_avg_df['recording'] = multisite_df['recording'] 


# In[31]:


region_avg_df.tail()


# In[ ]:


print('[INFO] plotting gamma waves for single mouse, pre and post...')
plt.figure(figsize=(10,5))


# **View quick statistics of our cleaned dataframe.**

# In[32]:


print('[INFO] dataframe description ...')
multisite_df.describe()


# **Decoding Helper Functions**

# In[ ]:


## data prep helper function
def dataSplit(df, x_target_columns, y_target_columns):
  # prepare data for decoding

  # pull out target data with columns  
  X=df[x_target_columns].to_numpy()
  y=multisite_df[y_target_columns]

  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

  return X_train, X_test, y_train, y_test;

## classifier helper function 
def svmModelBuilder(X_train,X_test,y_train, y_test, kernel='linear'):
  print("[INFO] building %s SVM classifier"%kernel)

  # build, fit and predict model
  clf = svm.SVC()
  clf.fit(X_train, y_train.values.ravel())
  print("[INFO] fit model for beta.")
  y_pred=clf.predict(X_test)
  print("[INFO] predicted model for beta.")

  # view metrics data
  print("\n[INFO] Model Metrics: \n")
  print(confusion_matrix(y_test,y_pred))
  print(classification_report(y_test,y_pred))

  return clf, y_pred;


## nested CV helper function
def nestedCVBuilder(clf, X, y, k_range=[50, 150, 300,500, 1000, 3000, 5000]):
  # build grid pipeline and run cross validation
  grid = GridSearchCV(clf, param_grid={'C': k_range})
  print("[INFO] built grid pipeline,running nested cv...")
  nested_cv_scores = cross_val_score(grid, X, y.values.ravel(), cv=5)
  print("[INFO completed nested cross validation.")
  print("Nested CV score: %.4f" % np.mean(nested_cv_scores * 100))

  return grid, nested_cv_scores;

# plot post and pre plots
def plotBrainwavesStatus(keyword=" "):
  plt.figure(figsize=(15,10))
  fig, (ax1, ax2) = plt.subplots(2,figsize=(10,10))
  fig.subplots_adjust(hspace=.4)
  ax1.set_title('%s pre-cocaine status'%keyword, fontsize=20)
  ax1.plot(multisite_df.loc[3600:5399, 'Time (s)'], multisite_df.loc[3600:5399, wave], label=wave+": pre",color='royalblue')
  ax2.set_title('%s post-cocaine status'%keyword, fontsize=20)
  ax2.plot(multisite_df.loc[0:1799, wave], label=wave+": post",color=post_color_dict[wave])


def plotBrainWaves(precolor, postcolor, keyword=" "):
  plt.figure(figsize=(10,5))

  for wave in pre_color_dict:
        if keyword in wave:
          plt.plot(multisite_df.loc[0:1799, wave], label=wave+": post",color=precolor)
          plt.plot(multisite_df.loc[3600:5399, 'Time (s)'], multisite_df.loc[3600:5399, wave], label=wave+": pre",color=postcolor)

          plt.title('Time Series Network, Mouse ID #1, recording 1, %s'%keyword)
          plt.xlabel('Time (s)')
          plt.ylabel('Wave Signal')
          plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
          plt.tight_layout()


# ## Decoding Gamma 
# Input X is the gamma brainwave data: `multisite_df[['PFC gamma', 'VTA gamma', 'BLA gamma', 'NAc gamma']]`,  
# target variable y is the `multisite_df['cocaine status num']`
# 
# The numerical cocaine status translates as follows:  { 0: post, 1: pre }
# 

# **Quick Plotting brainwave signals**

# Plot individual regions-

# In[106]:


print("[INFO] plotting regions...")
plotBrainWaves('royalblue', 'olive',keyword="PFC gamma")


# In[107]:


plotBrainWaves('royalblue', 'olive',keyword="NAc gamma")


# In[108]:


plotBrainWaves('royalblue', 'olive',keyword="BLA gamma")


# In[109]:


plotBrainWaves('royalblue', 'olive',keyword="VTA gamma")


# In[73]:


print("\n\n[INFO] plotting pair plot of gamma waves:\n")
sns.pairplot(multisite_df[['PFC gamma', 'VTA gamma', 'BLA gamma', 'NAc gamma']],diag_kind="kde")


# **Prepare input data**

# **Setup input X and target y**

# In[ ]:


# prepare the input data 
print('[INFO] preparing data for decoding gamma...')

# set X input array, 
# pull out the gamma columns from our dataframe and turn into numpy array
X=multisite_df[['PFC gamma', 'VTA gamma', 'BLA gamma', 'NAc gamma']].to_numpy()
print("X shape: ",X.shape)

# set y target variable 
y=multisite_df['cocaine status num']
print("y shape: ", y.shape)


# **Split datasets**
# 
# Training, testing and validation.

# In[ ]:


print("[INFO] splitting training dataset")


X_train, X_validate, y_train, y_validate = train_test_split(X, y, random_state=0, test_size = 0.20)

X_train.shape, X_validate.shape, y_train.shape, y_validate.shape


# **Create and fit the linear SVM classifier model**

# In[ ]:


print("[INFO] creating the svm linear model...")
# creater classifier model, svm
gamma_svm = svm.SVC(kernel='linear')
gamma_svm


# In[ ]:


# fit model with training data
gamma_svm.fit(X_train, y_train)
print("[INFO] fit model.")
# predict on validation data
gamma_y_pred=gamma_svm.predict(X_validate)
print("[INFO] predicted model.")


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print("[INFO] view metrics...")

print(confusion_matrix(y_validate,gma_y_pred))
print(classification_report(y_validate,gma_y_pred))


# In[ ]:


print("[INFO] setting up nested cross validation pipeline...")

# set up nested CV pipeline with grid 
from sklearn.model_selection import GridSearchCV, cross_val_score 
k_range =  [50, 150, 300,500, 1000, 3000, 5000]
gamma_grid = GridSearchCV(gamma_svm, param_grid={'C': k_range}, scoring='accuracy')
gamma_nested_cv_scores = cross_val_score(gamma_grid, X_train, y_train, cv=5)

print("[INFO] Finished CV.")
print("Nested CV score: %.4f" % np.mean(gamma_nested_cv_scores*100))


# In[ ]:


gamma_svm.coef_


# ## Decoding Beta

# In[110]:


print("[INFO] plotting regions for beta...")
plotBrainWaves('royalblue', 'olive',keyword="PFC beta")


# In[111]:


plotBrainWaves('royalblue', 'olive',keyword="NAc beta")


# In[112]:


plotBrainWaves('royalblue', 'olive',keyword="BLA beta")


# In[113]:


plotBrainWaves('royalblue', 'olive',keyword="VTA beta")


# In[ ]:


# prepare data for decoding
# pull out beta columns  
X=multisite_df[['PFC beta', 'VTA beta', 'BLA beta', 'NAc beta']].to_numpy()


# In[ ]:


y=multisite_df['cocaine status num']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[ ]:


beta_svm = svm.SVC()
beta_svm.fit(X_train, y_train)
print("[INFO] fit model for beta.")
y_pred=beta_svm.predict(X_test)
print("[INFO] predicted model for beta.")


# In[ ]:


print("[INFO] running nested cross validation on beta waves...")

k_range = [10, 15, 30, 50, 150, 300, 500, 1000, 1500, 3000, 5000]
beta_grid = GridSearchCV(beta_svm, param_grid={'C': k_range}, scoring='accuracy')
beta_nested_cv_scores = cross_val_score(beta_grid, X_train, y_train, cv=5)
print("[INFO completed nested cross validation.")
print("Nested CV score: %.4f" % np.mean(beta_nested_cv_scores*100))


# ## Decoding Theta

# In[114]:


print("[INFO] plotting regions...")
plotBrainWaves('royalblue', 'olive',keyword="PFC theta")


# In[115]:


plotBrainWaves('royalblue', 'olive',keyword="NAc theta")


# In[116]:


plotBrainWaves('royalblue', 'olive',keyword="BLA theta")


# In[117]:


plotBrainWaves('royalblue', 'olive',keyword="VTA theta")


# In[ ]:


# prepare data for decoding
# pull out theta columns  
X=multisite_df[['PFC theta', 'VTA theta', 'BLA theta', 'NAc theta',
                'reference wires', 'mouse id', 'Time (s)', 'Speed (cm/s)']].to_numpy()


# In[ ]:


y=multisite_df['cocaine status num']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[ ]:


print("[INFO] fitting classifier with theta waves...")

theta_svm = svm.SVC()
theta_svm.fit(X_train, y_train)
print("[INFO] fit model for theta.")
theta_y_pred=theta_svm.predict(X_test)
print("[INFO] predicted model for theta.")


# In[ ]:


print("[INFO] running cross validation on theta waves...")
theta_grid = GridSearchCV(theta_svm, param_grid={'C': k_range})
theta_nested_cv_scores = cross_val_score(theta_grid, X_train, y_train, cv=5)
print("[INFO completed nested cross validation.")
print("Nested CV score: %.4f" % np.mean(theta_nested_cv_scores * 100))


# ## Decoding with all regions and waves

# In[ ]:


all_regions_df = multisite_df.loc[:, ['PFC gamma', 'VTA gamma', 'BLA gamma', 'NAc gamma', 'PFC beta', 'VTA beta',
 'BLA beta', 'NAc beta', 'PFC theta', 'VTA theta', 'BLA theta', 'NAc theta', 'cocaine status num']]


# In[122]:


# full data decoding
plt.figure(figsize=(14,8))

for wave in pre_color_dict:
       plt.plot(multisite_df.loc[3600:5399, wave], label=wave,color=post_color_dict[wave])
 
plt.title('Time Series Network, Mouse ID #1, recording 1 pre-cocaine')
plt.xlabel('Time (s)')
plt.ylabel('Wave Signal')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()


plt.figure(figsize=(14,8))

for wave in post_color_dict:
       plt.plot(multisite_df.loc[0:1799, wave], label=wave,color=post_color_dict[wave])
 
plt.title('Time Series Network, Mouse ID #1, recording 1 post-cocaine')
plt.xlabel('Time (s)')
plt.ylabel('Wave Signal')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()


# In[ ]:


x_cols=['PFC gamma', 'VTA gamma', 'BLA gamma', 'NAc gamma', 'PFC beta', 'VTA beta',
 'BLA beta', 'NAc beta', 'PFC theta', 'VTA theta', 'BLA theta', 'NAc theta']
y_cols=['cocaine status num']


# In[ ]:


X_train, X_test, y_train, y_test=dataSplit(all_regions_df, x_cols, y_cols)


# In[ ]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


clf, y_pred=svmModelBuilder(X_train,X_test,y_train, y_test)


# In[ ]:


grid, nested_cv_scores=nestedCVBuilder(clf,X_train, y_train)


# ## Decoding by Regions
# 

# **PFC Region**

# In[123]:


# region based decoding
pfc_region= multisite_df.filter(regex=("PFC*"))
pfc_region['cocaine status']= multisite_df['cocaine status']
pfc_region.head()


# In[124]:


pfc_region.boxplot(figsize=(10,10))


# In[ ]:


x_cols=['PFC gamma', 'PFC beta', 'PFC theta']
y_cols=['cocaine status']
pfc_region['cocaine status']= multisite_df['cocaine status num']
X_train, X_test, y_train, y_test=dataSplit(pfc_region, x_cols, y_cols)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


clf, y_pred=svmModelBuilder(X_train,X_test,y_train, y_test)


# In[ ]:


grid, nested_cv_scores=nestedCVBuilder(clf,X_train, y_train)


# In[ ]:



vta_region= multisite_df.filter(regex=("VTA*"))
vta_region['cocaine status']= multisite_df.loc[:,'cocaine status']


# In[ ]:



nac_region= multisite_df.filter(regex=("NAc*"))
nac_region['cocaine status']= multisite_df.loc[:,'cocaine status']


# In[ ]:


bla_region= multisite_df.filter(regex=("BLA*"))
bla_region['cocaine status']= multisite_df.loc[:,'cocaine status']

