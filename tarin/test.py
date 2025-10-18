import numpy
import matplotlib.pyplot as  plt
from sklearn.metrics import r2_score
x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

plt.scatter(x, y)
plt.show()
#split into train and test
train_x=x[:80]
train_y=y[:80]
test_x = x[80:]
test_y = y[80:]
mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

myline = numpy.linspace(0, 6, 100)

plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
r2 = r2_score(train_y, mymodel(train_x))
print(r2)
plt.show()
#now bring the seting set
r2 = r2_score(test_y, mymodel(test_x))
print(r2)