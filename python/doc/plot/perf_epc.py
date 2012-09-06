#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Tutorial for plotting an EPC curve
"""

import bob
import numpy
from matplotlib import pyplot

dev_pos = numpy.random.normal(1,1,100)
dev_neg = numpy.random.normal(-1,1,100)
test_pos = numpy.random.normal(0.9,1,100)
test_neg = numpy.random.normal(-1.1,1,100)
npoints = 100
bob.measure.plot.epc(dev_neg, dev_pos, test_neg, test_pos, npoints, color=(0,0,0), linestyle='-')
pyplot.grid(True)
pyplot.title('EPC')
