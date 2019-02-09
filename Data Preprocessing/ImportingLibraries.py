#importing lib.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("/home/vivek/Desktop/Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values    
