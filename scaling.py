import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scaled=StandardScaler()
df=pd.read_csv("mldata.csv")

x=df[["Weight","Volume"]]
scaledx=scaled.fit_transform(x)
y=df["CO2"]
print(scaled)
#predict
regre=linear_model.LinearRegression()
regre.fit(scaledx,y)
scaled=scaled.transform([[2300,1.3]])
predected=regre.predict([scaled[0]])
print(predected)