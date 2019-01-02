#missing data and replacement by median & frequent no.
"""
Created on Sat Dec 15 16:20:38 2018

@author: vivek
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv("/home/vivek/Desktop/Data.csv")
X = dataset.iloc[:,:-1].values
Z = dataset.iloc[:,:-1].values
from sklearn.preprocessing import Imputer
impx = Imputer(missing_values='NaN', strategy='median',axis=0)
impx = impx.fit(X[:,1:3])
X[:,1:3] = impx.transform(X[:,1:3])
impz = Imputer(missing_values='NaN', strategy='most_frequent',axis=0)
impz = impz.fit(X[:,1:3])
Z[:,1:3] = impz.transform(Z[:,1:3])
print("Sai Naath Hotel Near Gwalior Station room no. 111")
