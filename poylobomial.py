import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
# from scipy.stats import Polynomial
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
mymodel=np.poly1d(np.polyfit(x,y,3))
myline=np.linspace(1,22,100)

plt.scatter(x, y)
plt.plot(myline,mymodel(myline))
plt.show()
#r squared
"""It is important to know how well the relationship between the values of the x- and y-axis is, if there are no relationship the polynomial regression can not be used to predict anything.

The relationship is measured with a value called the r-squared."""
print(r2_score(y,mymodel(x)))