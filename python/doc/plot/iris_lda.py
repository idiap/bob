#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Sat 24 Mar 2012 18:51:21 CET 

"""The Iris Flower Recognition using Linear Discriminant Analysis and Bob.
"""

import bob
import numpy
from matplotlib import pyplot

# Training is a 3-step thing
data = bob.db.iris.data()
trainer = bob.trainer.FisherLDATrainer()
machine, eigen_values = trainer.train(data.values())

# A simple way to forward the data
output = {}
for key in data.keys(): output[key] = machine(data[key])

# Here starts the plotting
pyplot.hist(output['setosa'][:,0], bins=8, 
    color='green', label='Setosa', alpha=0.5) 
pyplot.hist(output['versicolor'][:,0], bins=8,
    color='blue', label='Versicolor', alpha=0.5)
pyplot.hist(output['virginica'][:,0], bins=8, 
    color='red', label='Virginica', alpha=0.5)

# This is just some decoration...
pyplot.legend()
pyplot.grid(True)
pyplot.axis([-3,+3,0,20])
pyplot.title("Iris Plants / 1st. LDA component")
pyplot.xlabel("LDA[0]")
pyplot.ylabel("Count")
