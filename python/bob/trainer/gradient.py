#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Sun  2 Jun 16:09:29 2013 

"""Utilities for checking gradients.
"""

import numpy

def estimate(f, x, epsilon=1e-4, args=()):
  """Estimate the gradient for a given callable f

  Suppose you have a function :math:`f'(x)` that purportedly computes
  :math`\frac{\partial f(x)}{\partial x}`. You'd like to check if :math:`f'(x)`
  is outputting correct derivative values. You can then use this function to
  estimate the gradient around a point and compare it to the output of
  :math:`f'(x)`. The estimation can have a precision of up to a few decimal
  houses.

  Imagine a random value for :math:`x`, called :math:`x_t` (for test). Now
  imagine you modify one of the elements in :math:`x_t` so that
  :math:`x_{t+\epsilon}` has that element added with a small (positive) value
  :math:`\epsilon` and :math:`x_{t-\epsilon}` has the same value subtracted.

  In this case, one can use a truncated Taylor expansion of the derivative
  to calculate the approximate supposed value:

  .. math::
    f'(x_t) \sim \frac{f(x_{t+\epsilon}) - f(x_{t-\epsilon})}{2\epsilon}

  The degree to which these two values should approximate each other will
  depend on the details of :math:`f(x)`. But assuming :math:`\epsilon =
  10^{-4}`, you’ll usually find that the left- and right-hand sides of the
  above will agree to at least 4 significant digits (and often many more).

  Keyword arguments:

  f
    The function which you'd like to have the gradient estimated for.

  x
    The input to ``f``. This must be the first parameter ``f`` receives. If
    that is not the case, you must write a wrapper around ``f`` so it does the
    parameter inversion and provide that wrapper instead.

    If f expects a multi-dimensional array, than this entry should be a
    :py:class:`numpy.ndarray` with as many dimensions as required for f.

  precision
    The epsilon step

  args (optional)
    Extra arguments (a tuple) to ``f``

  This function returns the estimated value for :math:`f'(x)` given ``x``.

  .. note::

    Gradient estimation is a powerful tool for testing if a function is
    correctly computing the derivative of another function, but can be quite
    slow. It therefore is not a good replacement for writing specific code that
    can compute the derivative of ``f``.
  """
  epsilon = 1e-4

  if isinstance(x, numpy.ndarray):

    retval = numpy.ndarray(x.shape, dtype=x.dtype)
    for k in range(x.size):
      xt_plus = x.copy()
      xt_plus.flat[k] += epsilon
      xt_minus = x.copy()
      xt_minus.flat[k] -= epsilon
      retval.flat[k] = (f(xt_plus,*args) - f(xt_minus,*args)) / (2*epsilon)
    return retval
     
  else: # x is scalar
    return (f(x+epsilon, *args) - f(x-epsilon, *args)) / (2*epsilon)
