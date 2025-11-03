import pandas as pd
import os
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import warnings
os.chdir(os.path.dirname(__file__)) 
df=pd.read_csv("salaries.csv")
print(df)
d={"UK":0,"USA":1,"N":2}
df['Nationality'] = df['Nationality'].map(d)
d={"YES":1,"NO":0}
df["Go"]=df["Go"].map(d)
features=["Age","Experience","Rank","Nationality"]

x=df[features]
y=df["Go"]
print(x)
print(y)
dtree=DecisionTreeClassifier()
dtree=dtree.fit(x,y)
tree.plot_tree(dtree,feature_names=features )
print(dtree.predict([[40, 10, 7, 1]]))
warnings.filterwarnings("ignore")