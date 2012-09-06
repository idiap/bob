import bob
import numpy
from matplotlib import pyplot

# Training is a 3-step thing
data = bob.db.iris.data()
trainer = bob.trainer.FisherLDATrainer()
machine, eigen_values = trainer.train(data.values())

# A simple way to forward the data
output = {}
for key in data.keys(): output[key] = data[key].foreach(machine)

# From bob.io.Arrayset => numpy.ndarray 2D/float64
for key, value in output.iteritems():
  output[key] = numpy.vstack(value)

# Performance
negatives = numpy.vstack([output['setosa'], output['versicolor']])[:,0]
positives = output['virginica'][:,0]

# Plot ROC curve
bob.measure.plot.roc(negatives, positives)
pyplot.xlabel("False Virginica Rejection (%)")
pyplot.ylabel("False Virginica Acceptance (%)")
pyplot.title("ROC Curve for Virginica Classification")
pyplot.grid()
pyplot.axis([0, 15, 0, 5]) #xmin, xmax, ymin, ymax
