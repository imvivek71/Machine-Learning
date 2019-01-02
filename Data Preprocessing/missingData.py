# Missing data & solution by mean
"""
Created on Sat Dec 15 16:11:29 2018

@author: vivek
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("/home/vivek/Desktop/Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values    

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
