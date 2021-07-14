#!/usr/bin/env python
# coding: utf-8

# # Cocaine Connectivity Data PCA
# Description of data:  
# Time (s): 0 ~ 1799 per recording
# 
# 

# In[185]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


from IPython.core import display as ICD
import pandas as pd
import glob, os
import seaborn as sns
from sklearn import svm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[187]:


print("[INFO] loading data...\n")

#loading csv file 
multisite_df = pd.read_csv('drive/My Drive/Projects/pilot_mouse_connectivity/cocaine_two_mice/Spectrogram_data/updated_data/multisite_averaged_data.csv')

ICD.display(multisite_df.head())


# In[188]:


print("[INFO] dropping 'Unnamed: 0' column...\n" )
multisite_df.drop("Unnamed: 0", axis=1,inplace=True)

ICD.display(multisite_df.head())


# In[189]:


print(['[INFO] getting information on the data...'])


# In[190]:


multisite_df.info()


# In[191]:


multisite_df.describe()


# In[192]:


print("[INFO] filling NaN values with average...\n")
multisite_df.fillna(multisite_df.mean(), inplace=True)
ICD.display(multisite_df.head())


# In[193]:


print("[INFO] scaling the region columns...\n")

# set new df
scaled_df = multisite_df.loc[:,:]

# set scare obj
scaler = MinMaxScaler()
scaled_df[['PFC gamma', 'VTA gamma', 'BLA gamma', 'NAc gamma', 
           'PFC beta', 'VTA beta', 'BLA beta', 'NAc beta', 'PFC theta',
           'VTA theta', 'BLA theta', 'NAc theta', 'reference wires']] = scaler.fit_transform(scaled_df[['PFC gamma', 'VTA gamma', 'BLA gamma', 'NAc gamma',
                                                                                                       'PFC beta', 'VTA beta', 'BLA beta', 'NAc beta', 'PFC theta',
                                                                                                       'VTA theta', 'BLA theta', 'NAc theta', 'reference wires']])
       
ICD.display(scaled_df.head())


# In[194]:


print('[INFO] setting the x input variable...')
x = scaled_df.loc[:, ['PFC gamma', 'VTA gamma', 'BLA gamma', 'NAc gamma', 'PFC beta',
       'VTA beta', 'BLA beta', 'NAc beta', 'PFC theta', 'VTA theta', 'BLA theta', 'NAc theta']]


# In[ ]:


def fit_pca(n_components):
  print('[INFO] setting the PCA model and fitting with x...\n')

  from sklearn.decomposition import PCA
  pca = PCA(n_components=n_components)
  principalComponents = pca.fit_transform(x)

  print("PCA model: \n%s \n\nPrincipal Components: \n%s"%(pca,principalComponents))
  
  print("\nPCA variance: ", pca.explained_variance_ratio_)

  return pca, principalComponents;


# In[ ]:


def make_principalComponent_df(principalComponents,n_components,multisite_df):

  print("\n[INFO] making a datafame with principal compenents...\n")

  pc_list=[]

  # set the df column names
  for i in range(1,n_components+1):
    col_name = "principal component %i"%i
    pc_list.append(col_name)

  principal_df = pd.DataFrame(data = principalComponents, columns =pc_list)
  ICD.display(principal_df.describe())

  print("\n[INFO] combining PC dataframe and the target...\n")
  principal_df = pd.concat([principal_df, multisite_df[['cocaine status']]], axis = 1)
  ICD.display(principal_df.describe())

  return principal_df;


# In[ ]:


def visualize_heatmap(n_components,cmap):
  
  print('\n[INFO] visualizing heatmap...')

  y_labelsA = []
  y_labelsB = []

  title="%i-Component PCA"%n_components
  for i in range(1,n_components+1):

    label_name="PC %i"%i
    y_labelsA.append(i-1)
    y_labelsB.append(label_name)

  plt.matshow(pca.components_,cmap=cmap)
  plt.yticks(y_labelsA, y_labelsB, fontsize=10)
  plt.colorbar()
  #plt.title(title)
  plt.xticks(range(len(['PFC gamma', 'VTA gamma', 'BLA gamma', 'NAc gamma', 'PFC beta',
        'VTA beta', 'BLA beta', 'NAc beta', 'PFC theta', 'VTA theta', 'BLA theta', 'NAc theta'])),['PFC gamma', 'VTA gamma', 'BLA gamma', 'NAc gamma', 'PFC beta',
        'VTA beta', 'BLA beta', 'NAc beta', 'PFC theta', 'VTA theta', 'BLA theta', 'NAc theta'],rotation=65,ha='left')
  plt.tight_layout()
  plt.show()


# ## Running 2 component PCA 

# In[199]:


pca2, principalComponents2 = fit_pca(n_components=2)
pca2_df = make_principalComponent_df(principalComponents2, 2, multisite_df)
visualize_heatmap(n_components=2,cmap='RdPu')


# In[200]:


print('[INFO] visualizing principal components...')


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Principal Component Analysis', fontsize = 20)
targets = ['post', 'pre']
colors = ['indigo', 'deeppink']
for target, color in zip(targets,colors):
    indicesToKeep = pca2_df['cocaine status'] == target
    ax.scatter(pca2_df.loc[indicesToKeep, 'principal component 1']
               , pca2_df.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# ## Running 4 component PCA

# In[201]:


pca4, principalComponents4 = fit_pca(n_components=4)
pca4_df = make_principalComponent_df(principalComponents4, 4, multisite_df)
visualize_heatmap(n_components=4,cmap='twilight_shifted')

