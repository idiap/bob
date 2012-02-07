from ._cxx import *
from . import plot
from . import load
import numpy

def mse (estimation, target):
  """Calculates the mean square error between a set of outputs and target
  values using the following formula:

  .. math::

    MSE(\hat{\Theta}) = E[(\hat{\Theta} - \Theta)^2]

  Estimation (:math:`\hat{\Theta}`) and target (:math:`\Theta`) are supposed to
  have 2 dimensions. Different examples are organized as rows while different
  features in the estimated values or targets are organized as different
  columns.
  """
  return numpy.mean((estimation - target)**2, 0)

def rmse (estimation, target):
  """Calculates the root mean square error between a set of outputs and target
  values using the following formula:

  .. math::

    RMSE(\hat{\Theta}) = \sqrt(E[(\hat{\Theta} - \Theta)^2])

  Estimation (:math:`\hat{\Theta}`) and target (:math:`\Theta`) are supposed to
  have 2 dimensions. Different examples are organized as rows while different
  features in the estimated values or targets are organized as different
  columns.
  """
  return numpy.sqrt(mse(estimation, target))

def relevance (input, machine):
  """Calculates the relevance of every input feature to the estimation process
  using the following definition from:

  Neural Triggering System Operating on High Resolution Calorimetry
  Information, Anjos et al, April 2006, Nuclear Instruments and Methods in
  Physics Research, volume 559, pages 134-138 

  .. math::

    R(x_{i}) = |E[(o(x) - o(x|x_{i}=E[x_{i}]))^2]|

  In other words, the relevance of a certain input feature **i** is the change
  on the machine output value when such feature is replaced by its mean for all
  input vectors. For this to work, the `input` parameter has to be a 2D array
  with features arranged column-wise while different examples are arranged
  row-wise.
  """
  
  o = machine(input)
  i2 = input.copy()
  retval = numpy.ndarray((input.shape[1],), 'float64')
  retval.fill(0)
  for k in range(input.shape[1]):
    i2[:,:] = input #reset
    i2[:,k] = numpy.mean(input[:,k])
    retval[k] = (mse(machine(i2), o).sum())**0.5

  return retval

__all__ = dir()
