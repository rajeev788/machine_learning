import numpy
from sklearn import metrics
import matplotlib.pyplot as plt
a=numpy.random.binomial(1,0.9,size=1000)
predicted = numpy.random.binomial(1, 0.9, size = 1000)
confusion_matrix=metrics.confusion_matrix(a,predicted)
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,display_labels=[0,1])
cm_display.plot()
plt.show()