"""
# Import packages
"""

import pandas as pd
import glob, os 
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np
import time

from sklearn import preprocessing


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

##print(os.getcwd())
path="C:\\Users\\19802\\Documents\\nibl\\mouse_data\\Sucrose_infusions_dataframe_Eric.csv" #set path to file
data=pd.read_csv(path) # read in as dataframe

Y=data['Reward']
le = preprocessing.LabelEncoder()
le.fit(Y)
y_train_enc = le.transform(Y)

delta_bla=data[["BLA_delta", "MouseId", 'Time(s)']] # get target features
delta_bla=pd.get_dummies(delta_bla, columns=["MouseId"]) # one hot encoding on mouse ID
db5_X=delta_bla.to_numpy() # convert data to numpy array
db5_y=y_train_enc
db5_y=pd.to_numeric(db5_y, errors='coerce')

# Number of random trials
NUM_TRIALS = 30

# Set up possible values of parameters to optimize over
p_grid = {"C": [1, 10, 100],
          "gamma": [.01, .1]}
# We will use a Support Vector Classifier with "rbf" kernel
svm = SVC(kernel="rbf")
# Arrays to store scores
nested_scores = []
