from ._measure import *
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

def recognition_rate(cmc_scores):
  """Calculates the recognition rate from the given input, which is identical to the rank 1 (C)MC value.
  The input has a specific format, which is a list of two-element tuples.
  Each of the tuples contains the negative and the positive scores for one test item.
  To read the lists from score files in 4 or 5 column format, please use the bob.measure.load.cmc_four_column or bob.measure.load.cmc_five_column function.

  The recognition rate is defined as the number of test items,
  for which the positive score is greater than or equal to all negative scores,
  divided by the number of all test items.
  If several positive scores for one test item exist, the *highest* score is taken.
  """
  correct = 0.
  for neg, pos in cmc_scores:
    # get the maximum positive score for the current probe item
    # (usually, there is only one positive score, but just in case...)
    max_pos = numpy.max(pos)
    # check if the positive score is smaller than all negative scores
    if (neg <= max_pos).all():
      correct += 1

  # return relative number of
  return correct / float(len(cmc_scores))

def cmc(cmc_scores):
  """Calculates the cumulative match characteristic (CMC) from the given input.
  The input has a specific format, which is a list of two-element tuples.
  Each of the tuples contains the negative and the positive scores for one test item.
  To read the lists from score files in 4 or 5 column format, please use the bob.measure.load.cmc_four_column or bob.measure.load.cmc_five_column function.

  For each test item the probability that the rank r of the positive score is calculated.
  The rank is computed as the number of negative scores that are higher than the positive score.
  If several positive scores for one test item exist, the *highest* positive score is taken.
  The CMC finally computes, how many test items have rank r or higher.
  """
  # compute MC
  match_characteristic = numpy.zeros((max([len(neg) for (neg,pos) in cmc_scores])+1,), numpy.int)
  for neg, pos in cmc_scores:
    # get the maximum positive score for the current probe item
    # (usually, there is only one positive score, but just in case...)
    max_pos = numpy.max(pos)
    # count the number of negative scores that are higher than the best positive score
    index = numpy.count_nonzero(neg > max_pos)
    match_characteristic[index] += 1

  # cumulate
  probe_count = float(len(cmc_scores))
  cumulative_match_characteristic = numpy.ndarray(match_characteristic.shape, numpy.float64)
  count = 0.
  for i in range(match_characteristic.shape[0]):
    count += match_characteristic[i]
    cumulative_match_characteristic[i] = count / probe_count

  return cumulative_match_characteristic


__all__ = dir()
