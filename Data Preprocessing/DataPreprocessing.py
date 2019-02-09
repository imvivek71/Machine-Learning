# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Importing the lib..

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset

dataset = pd.read_csv('/home/vivek/Desktop/Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#Taking care of missing data

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
x[:,1:3] = imputer.fit_transform(x[:,1:3])

# Encoding Categorial Data & introduction of Dummy encoding onehotencoder

from sklearn.preprocessing import LabelEncoder,  OneHotEncoder 
x[:, 0]=LabelEncoder().fit_transform(x[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x=  onehotencoder.fit_transform(x).toarray()

y= LabelEncoder().fit_transform(y)

# Splitting the datset into training set and test set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=0)

# Feature Scaling of data

from sklearn.preprocessing import StandardScaler
sc_x =  StandardScaler()
x_train =sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
