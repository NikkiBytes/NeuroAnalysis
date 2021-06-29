# -- import packages --
import pandas as pd
import numpy as np 
import time
from matplotlib import pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, KFold,  cross_val_score
from numpy import mean,std
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class SVM:
    