#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri  7 Jun 08:59:24 2013 

"""Test cost functions
"""

import numpy
import math

from .. import SquareError, CrossEntropyLoss, gradient
from ...machine import LogisticActivation, IdentityActivation

def test_square_error():
  
  op = SquareError(IdentityActivation())
  x = numpy.random.rand(10) #10 random numbers between 0 and 1
  y = numpy.random.rand(10) #10 random numbers between 0 and 1

  # go for an exact match
  for p,q in zip(x,y):
    expected = 0.5*math.pow(p-q,2)
    assert op.f(p,q) == expected, 'SquareError does not perform as expected %g != %g' % (op.f(p,q), expected)

def test_square_error_derivative():
  
  op = SquareError(IdentityActivation())
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

def test_square_error_error():
  
  act = LogisticActivation()
  op = SquareError(act)
  x = numpy.random.rand(10) #10 random numbers between 0 and 1
  y = numpy.random.rand(10) #10 random numbers between 0 and 1

  # go for an exact match
  for p,q in zip(x,y):
    expected = p*(1-p)*(p-q)
    assert op.error(p,q) == expected, 'SquareError error does not perform as expected %g != %g' % (op.error(p,q), expected)

def test_cross_entropy():
  
  op = CrossEntropyLoss(LogisticActivation())
  x = numpy.random.rand(10) #10 random numbers between 0 and 1
  y = numpy.random.rand(10) #10 random numbers between 0 and 1

  # go for an exact match
  for p,q in zip(x,y):
    expected = -q*math.log(p) - (1-q)*math.log(1-p)
    assert op.f(p,q) == expected, 'CrossEntropyLoss does not perform as expected %g != %g' % (op.f(p,q), expected)

def test_cross_entropy_derivative():
  
  op = CrossEntropyLoss(LogisticActivation())
  x = numpy.random.rand(10) #10 random numbers between 0 and 1
  y = numpy.random.rand(10) #10 random numbers between 0 and 1

  # go for an exact match
  for p,q in zip(x,y):
    expected = (p-q)/(p*(1-p))
    assert op.f_prime(p,q) == expected, 'CrossEntropyLoss derivative does not perform as expected %g != %g' % (op.f(p,q), expected)

  # go for approximation
  for p,q in zip(x,y):
    reldiff = abs((op.f_prime(p,q)-gradient.estimate(op.f,p,args=(q,))) / op.f_prime(p,q))
    assert reldiff < 1e-3, 'SquareError derivative and estimation do not match to 10^-4: |%g-%g| = %g' % (op.f_prime(p,q), gradient.estimate(op.f,p,args=(q,)), reldiff)

def test_square_error_equality():

  op1 = SquareError(IdentityActivation())
  op2 = SquareError(IdentityActivation())

  assert op1 == op2

def test_cross_entropy_equality():

  op1 = CrossEntropyLoss(IdentityActivation())
  op2 = CrossEntropyLoss(IdentityActivation())

  assert op1 == op2

def test_cross_entropy_error_with_logistic():
  
  act = LogisticActivation()
  op = CrossEntropyLoss(act)
  x = numpy.random.rand(10) #10 random numbers between 0 and 1
  y = numpy.random.rand(10) #10 random numbers between 0 and 1

  # go for an exact match
  for p,q in zip(x,y):
    expected = p-q
    assert op.error(p,q) == expected, 'CrossEntropyLoss+LogisticActivation error does not perform as expected %g != %g' % (op.error(p,q), expected)

def test_cross_entropy_error_without_logistic():
  
  act = IdentityActivation()
  op = CrossEntropyLoss(act)
  x = numpy.random.rand(10) #10 random numbers between 0 and 1
  y = numpy.random.rand(10) #10 random numbers between 0 and 1

  # go for an exact match
  for p,q in zip(x,y):
    expected = (p-q)/(p*(1-p))
    assert op.error(p,q) == expected, 'CrossEntropyLoss+IdentityActivation error does not perform as expected %g != %g' % (op.error(p,q), expected)

def test_cross_entropy_activation_detection():

  op = CrossEntropyLoss(LogisticActivation())
  assert op.logistic_activation

  op = CrossEntropyLoss(IdentityActivation())
  assert op.logistic_activation == False
