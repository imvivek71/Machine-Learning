#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 14:13:34 2019

@author: vivek
"""


# Importing the dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset

dataset = pd.read_csv('/home/vivek/Desktop/Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# Splitting the datset into training set and test set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(x,y, test_size = 1/3, random_state=0)

# Feature Scaling 

"""
from sklearn.preprocessing import StandardScaler
sc_x =  StandardScaler()
x_train =sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""
# Fitting simple linear regression 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

# Visulaising the training set results

plt.scatter(x_train,y_train, color ='red')
plt.plot(x_train,regressor.predict(x_train), color = 'blue')
plt.title('Experience vs Salary(Training Set)')
plt.xlabel('Experience') 
plt.ylabel('Salary')
plt.show()

# Visulaising the test set results

plt.scatter(x_test,y_test, color ='red')
plt.plot(x_train,regressor.predict(x_train), color = 'blue')
plt.title('Experience vs Salary(Test Set)')
plt.xlabel('Experience') 
plt.ylabel('Salary')
plt.show()
