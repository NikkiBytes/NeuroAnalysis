#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -- Packages --
import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('../scripts/preprocessing_module')
import BBXReport

# -- Display Settings -- 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
get_ipython().run_line_magic('pylab', 'inline')
plt.rcParams["figure.figsize"] = (15,12)


# # BBX Summary Report 
# ![](../../data/pngs/sub-001_ses-1_anat.png)

# In[2]:


BBXObj = BBXReport.BBXReport()


# In[3]:


BBXObj.bbx_dcm_report() # dicom report


# In[4]:


BBXObj.bids_report() # bids report


# ---

# In[ ]:




