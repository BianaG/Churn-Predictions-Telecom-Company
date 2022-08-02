# Churn-Predictions-Telecom-Company
The telecom operator would like to be able to forecast their churn of clients. If it's discovered that a user is planning to leave, they will be offered promotional codes and special plan options. The marketing team has collected some of their clientele's personal data, including information about their plans and contracts.

The libraries that had been used in this task is:

#### --Data calculation and analysis libraries--

import pandas as pd

import numpy as np

import math as m

import sidetable as stb

from datetime import datetime, date

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Spliting data function
from sklearn.model_selection import train_test_split

# shuffle function
from sklearn.utils import shuffle

# Data engineering libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer

# Evaluation metrics 
from sklearn.metrics import roc_auc_score, f1_score

# Classification models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

# A hyperparameter optimization framework to automate hyperparameter search
import optuna

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
