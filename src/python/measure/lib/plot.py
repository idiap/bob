#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Chakka Murali Mohan, Trainee, IDIAP Research Institute, Switzerland.
# Mon 23 May 2011 14:36:14 CEST

"""Methods to plot error analysis figures such as ROC, EPC and DET"""

def roc(negatives, positives, npoints=100, **kwargs):
  """Plots Receiver Operating Charactaristic (ROC) curve.

  This method will call matplotlib to plot the ROC curve(s) for a system which
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
    a blitz array of negative class scores in float64 format or several, packed
    in a list

  positives 
    a blitz array of positive class scores in float64 format or several, packed
    in a list

  npoints
    number of points to use when drawing the ROC curve

  kwargs
    a dictionary of extra plotting parameters, that is passed directly to
    matplotlib.pyplot.plot().
  
  .. note::
  
    This function does not initiate and save the figure instance, it only
    issues the plotting command. You are the responsible for setting up and
    saving the figure as you see fit.  

  Return value is a list of lines that were added as defined by the
  matplotlib.pyplot.plot() command.
  """
  
  try:
    import matplotlib.pyplot as mpl
  except ImportError:
    print("Cannot import matplotlib. This package is not essential, but required if you wish to use the plotting functionality.")
    raise

  if not isinstance(negatives, (tuple, list)): negatives = (negatives,)
  if not isinstance(positives, (tuple, list)): positives = (positives,)

  if len(negatives) != len(positives):
    raise RuntimeError, "Length of negatives (%d) != positives (%d)" % \
        (len(negatives), len(positives))
  
  from . import roc as calc

  out = [calc(negatives[k], positives[k], npoints) for k in
      range(len(negatives))]
  x = [out[k][0,:] for k in range(len(negatives))]
  y = [out[k][1,:] for k in range(len(negatives))]

  return mpl.plot(x, y, **kwargs)

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
    blitz array of negative class scores on development set in float64 format,
    or a list of those
  
  dev_positives
    blitz array of positive class scores on development set in float64 format,
    or a list of those

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

  Return value is a list of lines that were added as defined by the
  matplotlib.pyplot.plot() command.
  """

  try:
    import matplotlib.pyplot as mpl
  except ImportError:
    print("Cannot import matplotlib. This package is not essential, but required if you wish to use the plotting functionality.")
    raise
  
  if not isinstance(dev_negatives, (tuple, list)): 
    dev_negatives = (dev_negatives,)
  if not isinstance(dev_positives, (tuple, list)): 
    dev_positives = (dev_positives,)

  if not isinstance(test_negatives, (tuple, list)): 
    test_negatives = (test_negatives,)
  if not isinstance(test_positives, (tuple, list)): 
    test_positives = (test_positives,)

  if len(dev_negatives) != len(dev_positives):
    raise RuntimeError, "Length of dev. negatives (%d) != positives (%d)" % \
        (len(dev_negatives), len(dev_positives))
  if len(test_negatives) != len(test_positives):
    raise RuntimeError, "Length of test negatives (%d) != positives (%d)" % \
        (len(test_negatives), len(test_positives))
  if len(test_negatives) != len(dev_negatives):
    raise RuntimeError, \
        "Length of test negatives (%d) != dev. negatives (%d)" % \
        (len(test_negatives), len(dev_negatives))
  
  from . import epc as calc

  out = [calc(dev_negatives[k], dev_positives[k], test_negatives,
    test_positives, npoints) for k in range(len(dev_negatives))]

  x = [out[k][0,:] for k in range(len(dev_negatives))]
  y = [out[k][1,:] for k in range(len(dev_negatives))]

  return mpl.plot(x, y, **kwargs)

def det(negatives, positives, npoints=100, 
    limits=("0.001", "99.999", "0.001", "99.999"), **kwargs):
  """Plots Detection Error Trade-off (DET) curve as defined in the paper:

  Martin, A., Doddington, G., Kamm, T., Ordowski, M., & Przybocki, M. (1997).
  The DET curve in assessment of detection task performance. Fifth European
  Conference on Speech Communication and Technology (pp. 1895-1898). Available:
  http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.117.4489&rep=rep1&type=pdf 

  This method will call matplotlib to plot the DET curve(s) for a system which
  contains a particular set of negatives (impostors) and positives (clients)
  scores. We use the standard matplotlib.pyplot.plot() command. All parameters
  passed with exception of the four first parameters of this method will be
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
    
  options
    dictionary containing optional parameters
    options dictionary items:

  limits
    tuple containting X & Y axis limits (as strings), default limits are 
    ("0.001", "99.999", "0.001", "99.999"). The values contained in the limits
    tuple must be one of the following:

      "0.001", "0.002", "0.005", "0.01", "0.02", "0.05",
      "0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "40", "60", "80", "90",
      "95", "98", "99", "99.5", "99.8", "99.9", "99.95", "99.98", "99.99",
      "99.995", "99.998", "99.999"

    The order should be (min_x, max_x, min_y, max_y).

    .. note::

      This method is different from the others in which it receives a direct
      setting for the plot axis in the form of a "limits" tuple. This is the
      case because it is non-trivial do zoom-in/out a DET plot and using
      natural limits makes it simpler for the user.
    
  .. note::
  
    This function does not initiate and save the figure instance, it only
    issues the plotting commands. You are the responsible for setting up and
    saving the figure as you see fit.

  Return value is a list of lines that were added as defined by the
  matplotlib.pyplot.plot() command.
  """

  try:
    import matplotlib.pyplot as mpl
  except ImportError:
    print("Cannot import matplotlib. This package is not essential, but required if you wish to use the plotting functionality.")
    raise

  if not isinstance(negatives, (tuple, list)): negatives = (negatives,)
  if not isinstance(positives, (tuple, list)): positives = (positives,)

  if len(negatives) != len(positives):
    raise RuntimeError, "Length of negatives (%d) != positives (%d)" % \
        (len(negatives), len(positives))
 
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

  out = [calc(negatives[k], positives[k], npoints) for k in range(len(negatives))]

  x = [out[k][0,:] for k in range(len(negatives))]
  y = [out[k][1,:] for k in range(len(negatives))]

  retval = mpl.plot(x, y, **kwargs)

  # now the trick: we must plot the tick marks by hand using the PPNDF method
  fr_minIndex = desiredLabels.index(limits[0])
  fr_maxIndex = desiredLabels.index(limits[1])
  fa_minIndex = desiredLabels.index(limits[2])
  fa_maxIndex = desiredLabels.index(limits[3])

  # converts tick marks into DET scale (that is the scale for the plot)
  pticks = [ppndf(float(v)) for v in desiredTicks]

  ax = mpl.gca()

  # zooms in using the DET scale
  mpl.axis([pticks[fr_minIndex], pticks[fr_maxIndex], 
    pticks[fa_minIndex], pticks[fa_maxIndex]])

  # selectively sets the x and y ticks
  ax.set_xticks(pticks[fr_minIndex:fr_maxIndex])
  ax.set_xticklabels(desiredLabels[fr_minIndex:fr_maxIndex], 
      size='x-small', rotation='vertical')
  ax.set_yticks(pticks[fa_minIndex:fa_maxIndex])
  ax.set_yticklabels(desiredLabels[fa_minIndex:fa_maxIndex],
      size='x-small')

  return retval
