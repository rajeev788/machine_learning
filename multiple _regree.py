import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn import linear_model
# os.chdir(os.path.dirname(__file__)) 
data=pd.read_csv("mldata.csv")
x=data[['Weight','Volume']]
y=data['CO2']
regre=linear_model.LinearRegression()
regre.fit(x,y)
predicted=regre.predict([
    [2300,1300]
])
print(predicted)
# print(pd.DataFrame(data))