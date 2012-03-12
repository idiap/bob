#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Chakka Murali Mohan, Trainee, IDIAP Research Institute, Switzerland.
# Mon 23 May 2011 14:36:14 CEST

"""Methods to plot error analysis figures such as ROC, EPC and DET"""

def roc(negatives, positives, npoints=100, **kwargs):
  """Plots Receiver Operating Charactaristic (ROC) curve.

  This method will call matplotlib to plot the ROC curve for a system which
  contains a particular set of negatives (impostors) and positives (clients)
  scores. We use the standard matplotlib.pyplot.plot() command. All parameters
  passed with exeception of the three first parameters of this method will be
  directly passed to the plot command. If you wish to understand your options,
  look here:

  http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.plot

  The plot will represent the false-alarm on the vertical axis and the
  false-rejection on the horizontal axis.

  .. warning::  
    This method requires that matplotlib is installed and operational on the
    system executing the plot.

  Input arguments:

  negatives
    a blitz array of negative class scores in float64 format

  positives 
    a blitz array of positive class scores in float64 format

  npoints
    number of points to use when drawing the ROC curve

  kwargs
    a dictionary of extra plotting parameters, that is passed directly to
    matplotlib.pyplot.plot().
  
  .. note::
  
    This function does not initiate and save the figure instance, it only
    issues the plotting command. You are the responsible for setting up and
    saving the figure as you see fit.  

  Return value is the matplotlib line that was added as defined by the
  matplotlib.pyplot.plot() command.
  """

  try:
    import matplotlib.pyplot as mpl
  except ImportError:
    print("Cannot import matplotlib. This package is not essential, but required if you wish to use the plotting functionality.")
    raise
  
  from . import roc as calc
  out = calc(negatives, positives, npoints)
  return mpl.plot(100.0*out[0,:], 100.0*out[1,:], **kwargs)

def epc(dev_negatives, dev_positives, test_negatives, test_positives, 
    npoints=100, **kwargs):
  """Plots Expected Performance Curve (EPC) as defined in the paper:

  Bengio, S., Keller, M., Mariéthoz, J. (2004). The Expected Performance Curve.
  International Conference on Machine Learning ICML Workshop on ROC Analysis in
  Machine Learning, 136(1), 1963–1966. IDIAP RR. Available:
  http://eprints.pascal-network.org/archive/00000670/

  This method will call matplotlib to plot the EPC curve for a system which
  contains a particular set of negatives (impostors) and positives (clients)
  for both the development and test sets. We use the standard
  matplotlib.pyplot.plot() command. All parameters passed with exeception of
  the five first parameters of this method will be directly passed to the plot
  command. If you wish to understand your options, look here:

  http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.plot

  The plot will represent the minimum HTER on the vertical axis and the
  cost on the horizontal axis.

  .. warning::  
    This method requires that matplotlib is installed and operational on the
    system executing the plot.

  Input arguments:

  dev_negatives
    blitz array of negative class scores on development set in float64 format  

  dev_positives
    blitz array of positive class scores on development set in float64 format

  test_negatives
    blitz array of negative class scores on test set in float64 format, or a
    list of those
  
  test_positives
    blitz array of positive class scores on test set in float64 format, or a
    list of those
  
  npoints
    number of points to use when drawing the EPC curve

  kwargs
    a dictionary of extra plotting parameters, that is passed directly to
    matplotlib.pyplot.plot().
  
  .. note::
  
    This function does not initiate and save the figure instance, it only
    issues the plotting commands. You are the responsible for setting up and
    saving the figure as you see fit.  

  Return value is the matplotlib line that was added as defined by the
  matplotlib.pyplot.plot() command.
  """

  try:
    import matplotlib.pyplot as mpl
  except ImportError:
    print("Cannot import matplotlib. This package is not essential, but required if you wish to use the plotting functionality.")
    raise
  
  from . import epc as calc

  out = calc(dev_negatives, dev_positives, test_negatives, test_positives, 
      npoints)
  return mpl.plot(out[0,:], 100.0*out[1,:], **kwargs)

def det(negatives, positives, npoints=100, axisfontsize='x-small', **kwargs):
  """Plots Detection Error Trade-off (DET) curve as defined in the paper:

  Martin, A., Doddington, G., Kamm, T., Ordowski, M., & Przybocki, M. (1997).
  The DET curve in assessment of detection task performance. Fifth European
  Conference on Speech Communication and Technology (pp. 1895-1898). Available:
  http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.117.4489&rep=rep1&type=pdf 

  This method will call matplotlib to plot the DET curve(s) for a system which
  contains a particular set of negatives (impostors) and positives (clients)
  scores. We use the standard matplotlib.pyplot.plot() command. All parameters
  passed with exception of the three first parameters of this method will be
  directly passed to the plot command. If you wish to understand your options,
  look here:

  http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.plot

  The plot will represent the false-alarm on the vertical axis and the
  false-rejection on the horizontal axis.

  This method is strongly inspired by the NIST implementation for Matlab,
  called DETware, version 2.1 and available for download at the NIST website:

  http://www.itl.nist.gov/iad/mig/tools/

  .. warning::
    This method requires that matplotlib is installed and operational on the
    system executing the plot.

  Keyword parameters:

  positives
    numpy.array of positive class scores in float64 format
    
  negatives
    numpy.array of negative class scores in float64 format
    
  npoints
    number of points to use when drawing the EPC curve

  axisfontsize
    the size to be used by x/ytickables to set the font size on the axis

  kwargs
    a dictionary of extra plotting parameters, that is passed directly to
    matplotlib.pyplot.plot().

  .. note::
  
    This function does not initiate and save the figure instance, it only
    issues the plotting commands. You are the responsible for setting up and
    saving the figure as you see fit.

  .. note::

    If you wish to reset axis zooming, you must use the gaussian scale rather
    than the visual marks showed at the plot, which are just there for
    displaying purposes. The real axis scale is based on the
    bob.measure.ppndf() method. For example, if you wish to set the x and y
    axis to display data between 1% and 40% here is the recipe:

    .. code-block:: python

      import bob
      import matplotlib.pyplot as mpl
      bob.measure.plot.det(...) #call this as many times as you need
      #AFTER you plot the DET curve, just set the axis in this way:
      mpl.axis([bob.measure.ppndf(k/100.0) for k in (1, 40, 1, 40)])

    We provide a convenient way for you to do the above in this module. So,
    optionally, you may use the bob.measure.plot.det_axis() method like this:

    .. code-block:: python

      import bob
      bob.measure.plot.det(...)
      # please note we convert percentage values in det_axis()
      bob.measure.plot.det_axis([1, 40, 1, 40])

  Return value is the matplotlib line that was added as defined by the
  matplotlib.pyplot.plot() command.
  """

  try:
    import matplotlib.pyplot as mpl
  except ImportError:
    print("Cannot import matplotlib. This package is not essential, but required if you wish to use the plotting functionality.")
    raise

  # these are some constants required in this method
  desiredTicks = [
      "0.00001", "0.00002", "0.00005",
      "0.0001", "0.0002", "0.0005",
      "0.001", "0.002", "0.005",
      "0.01", "0.02", "0.05",
      "0.1", "0.2", "0.4", "0.6", "0.8", "0.9",
      "0.95", "0.98", "0.99",
      "0.995", "0.998", "0.999",
      "0.9995", "0.9998", "0.9999",
      "0.99995", "0.99998", "0.99999"
      ]

  desiredLabels = [
      "0.001", "0.002", "0.005",
      "0.01", "0.02", "0.05",
      "0.1", "0.2", "0.5",
      "1", "2", "5",
      "10", "20", "40", "60", "80", "90",
      "95", "98", "99",
      "99.5", "99.8", "99.9",
      "99.95", "99.98", "99.99",
      "99.995", "99.998", "99.999"
      ]

  # this will actually do the plotting
  from . import det as calc
  from . import ppndf

  out = calc(negatives, positives, npoints)
  retval = mpl.plot(out[0,:], out[1,:], **kwargs)

  # now the trick: we must plot the tick marks by hand using the PPNDF method
  pticks = [ppndf(float(v)) for v in desiredTicks]
  ax = mpl.gca() #and finally we set our own tick marks
  ax.set_xticks(pticks)
  ax.set_xticklabels(desiredLabels, size=axisfontsize)
  ax.set_yticks(pticks)
  ax.set_yticklabels(desiredLabels, size=axisfontsize)

  return retval

def det_axis(v, **kwargs):
  """Sets the axis in a DET plot.

  This method wraps the matplotlib.pyplot.axis() by calling
  bob.measure.ppndf() on the values passed by the user so they are meaningful
  in a DET plot as performed by bob.measure.plot.det().

  Keyword parameters:

  v 
    Python iterable (list or tuple) with the X and Y limits in the order (xmin,
    xmax, ymin, ymax). Expected values should be in percentage (between 0 and
    100%). If v is not a list or tuple that contains 4 numbers it is passed
    without further inspection to matplotlib.pyplot.axis().

  kwargs
    All remaining arguments will be passed to matplotlib.pyplot.axis() without
    further inspection.

  Returns whatever matplotlib.pyplot.axis() returns.
  """

  import logging
  
  try:
    import matplotlib.pyplot as mpl
  except ImportError:
    print("Cannot import matplotlib. This package is not essential, but required if you wish to use the plotting functionality.")
    raise

  from . import ppndf

  # treat input
  try:
    tv = list(v) #normal input
    if len(tv) != 4: raise IndexError
    tv = [ppndf(float(k)/100) for k in tv]
    cur = mpl.axis()

    # limits must be within bounds
    if tv[0] < cur[0]:
      logging.warn("Readjusting xmin: the provided value is out of bounds")
      tv[0] = cur[0]
    if tv[1] > cur[1]: 
      logging.warn("Readjusting xmax: the provided value is out of bounds")
      tv[1] = cur[1]
    if tv[2] < cur[2]: 
      logging.warn("Readjusting ymin: the provided value is out of bounds")
      tv[2] = cur[2]
    if tv[3] > cur[3]: 
      logging.warn("Readjusting ymax: the provided value is out of bounds")
      tv[3] = cur[3]

  except:
    tv = v

  ax = mpl.axis(tv, **kwargs)

  ax.set_xticklabels(desiredLabels, size='x-small')
  ax.set_yticklabels(desiredLabels, size='x-small')
  return ax
