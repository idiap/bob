#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Tutorial for plotting a ROC curve
"""

import bob
import numpy
from matplotlib import pyplot

positives = numpy.random.normal(1,1,100)
negatives = numpy.random.normal(-1,1,100)
npoints = 100
bob.measure.plot.roc(negatives, positives, npoints, color=(0,0,0), linestyle='-', label='test')
pyplot.grid(True)
pyplot.xlabel('FRR (%)')
pyplot.ylabel('FAR (%)')
pyplot.title('ROC')
