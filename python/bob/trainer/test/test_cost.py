#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri  7 Jun 08:59:24 2013 

"""Test cost functions
"""

import numpy
import math

from .. import SquareError, CrossEntropyLoss, gradient

def test_square_error():
  
  op = SquareError()
  x = numpy.random.rand(10) #10 random numbers between 0 and 1
  y = numpy.random.rand(10) #10 random numbers between 0 and 1

  # go for an exact match
  for p,q in zip(x,y):
    expected = 0.5*math.pow(p-q,2)
    assert op.f(p,q) == expected, 'SquareError does not perform as expected %g != %g' % (op.f(p,q), expected)

def test_square_error_derivative():
  
  op = SquareError()
  x = numpy.random.rand(10) #10 random numbers between 0 and 1
  y = numpy.random.rand(10) #10 random numbers between 0 and 1

  # go for an exact match
  for p,q in zip(x,y):
    expected = p-q
    assert op.f_prime(p,q) == expected, 'SquareError derivative does not perform as expected %g != %g' % (op.f(p,q), expected)

  # go for approximation
  for p,q in zip(x,y):
    absdiff = abs(op.f_prime(p,q)-gradient.estimate(op.f,p,args=(q,)))
    assert absdiff < 1e-4, 'SquareError derivative and estimation do not match to 10^-4: |%g-%g| = %g' % (op.f_prime(p,q), gradient.estimate(op.f,p,args=(q,)), absdiff)

def test_cross_entropy():
  
  op = CrossEntropyLoss()
  x = numpy.random.rand(10) #10 random numbers between 0 and 1
  y = numpy.random.rand(10) #10 random numbers between 0 and 1

  # go for an exact match
  for p,q in zip(x,y):
    expected = -q*math.log(p) - (1-q)*math.log(1-p)
    assert op.f(p,q) == expected, 'CrossEntropyLoss does not perform as expected %g != %g' % (op.f(p,q), expected)

def test_cross_entropy_derivative():
  
  op = CrossEntropyLoss()
  x = numpy.random.rand(10) #10 random numbers between 0 and 1
  y = numpy.random.rand(10) #10 random numbers between 0 and 1

  # go for an exact match
  for p,q in zip(x,y):
    expected = (p-q)/(p*(1-p))
    assert op.f_prime(p,q) == expected, 'CrossEntropyLoss derivative does not perform as expected %g != %g' % (op.f(p,q), expected)

  # go for approximation
  for p,q in zip(x,y):
    absdiff = abs(op.f_prime(p,q)-gradient.estimate(op.f,p,args=(q,)))
    assert absdiff < 1e-4, 'SquareError derivative and estimation do not match to 10^-4: |%g-%g| = %g' % (op.f_prime(p,q), gradient.estimate(op.f,p,args=(q,)), absdiff)

def test_square_error_equality():

  op1 = SquareError()
  op2 = SquareError()

  assert op1 == op2

def test_cross_entropy_equality():

  op1 = CrossEntropyLoss()
  op2 = CrossEntropyLoss()

  assert op1 == op2
