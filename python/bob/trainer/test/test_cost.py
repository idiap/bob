#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri  7 Jun 08:59:24 2013 

"""Test cost functions
"""

import numpy
import math

from .. import SquareError, CrossEntropyLoss
from ...machine import LogisticActivation, IdentityActivation
from . import gradient

def is_close(x, y, eps=1e-10):
  return (abs(x - y) < eps)

def rand_safe0(eps=2e-4):
  return numpy.random.rand()*(1-2*eps)+eps

def rand_safe(n, eps=2e-4):
  return numpy.random.rand(n)*(1-2*eps)+eps

def rand_safe2(n, p, eps=2e-4):
  return numpy.random.rand(n,p)*(1-2*eps)+eps

def rand_safe3(n, p, q, eps=2e-4):
  return numpy.random.rand(n,p,q)*(1-2*eps)+eps

def rand_safe4(n, p, q, r, eps=2e-4):
  return numpy.random.rand(n,p,q,r)*(1-2*eps)+eps

def test_square_error():
  
  op = SquareError(IdentityActivation())
  x = rand_safe(10) #10 random numbers between 0 and 1
  y = rand_safe(10) #10 random numbers between 0 and 1

  # go for an exact match
  for p,q in zip(x,y):
    expected = 0.5*math.pow(p-q,2)
    assert is_close(op.f(p,q), expected), 'SquareError does not perform as expected %g != %g' % (op.f(p,q), expected)

def test_square_error_derivative():
  
  op = SquareError(IdentityActivation())
  x = rand_safe(10) #10 random numbers between 0 and 1
  y = rand_safe(10) #10 random numbers between 0 and 1

  # go for an exact match
  for p,q in zip(x,y):
    expected = p-q
    assert is_close(op.f_prime(p,q), expected), 'SquareError derivative does not perform as expected %g != %g' % (op.f(p,q), expected)

  # go for approximation
  for p,q in zip(x,y):
    absdiff = abs(op.f_prime(p,q)-gradient.estimate(op.f,p,args=(q,)))
    assert absdiff < 1e-4, 'SquareError derivative and estimation do not match to 10^-4: |%g-%g| = %g' % (op.f_prime(p,q), gradient.estimate(op.f,p,args=(q,)), absdiff)

def test_square_error_error():
  
  act = LogisticActivation()
  op = SquareError(act)
  x = rand_safe(10) #10 random numbers between 0 and 1
  y = rand_safe(10) #10 random numbers between 0 and 1

  # go for an exact match
  for p,q in zip(x,y):
    expected = p*(1-p)*(p-q)
    assert is_close(op.error(p,q), expected), 'SquareError error does not perform as expected %g != %g' % (op.error(p,q), expected)

def test_cross_entropy():
  
  op = CrossEntropyLoss(LogisticActivation())
  x = rand_safe(10) #10 random numbers between 0 and 1
  y = rand_safe(10) #10 random numbers between 0 and 1

  # go for an exact match
  for p,q in zip(x,y):
    expected = -q*math.log(p) - (1-q)*math.log(1-p)
    assert is_close(op.f(p,q), expected), 'CrossEntropyLoss does not perform as expected %g != %g' % (op.f(p,q), expected)

def test_cross_entropy_derivative():
  
  op = CrossEntropyLoss(LogisticActivation())
  x = rand_safe(10, eps=1e-3) #10 random numbers between 0 and 1
  y = rand_safe(10, eps=1e-3) #10 random numbers between 0 and 1

  # go for an exact match
  for p,q in zip(x,y):
    expected = (p-q)/(p*(1-p))
    assert is_close(op.f_prime(p,q), expected), 'CrossEntropyLoss derivative does not perform as expected %g != %g' % (op.f(p,q), expected)

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
  x = rand_safe(10) #10 random numbers between 0 and 1
  y = rand_safe(10) #10 random numbers between 0 and 1

  # go for an exact match
  for p,q in zip(x,y):
    expected = p-q
    assert is_close(op.error(p,q), expected), 'CrossEntropyLoss+LogisticActivation error does not perform as expected %g != %g' % (op.error(p,q), expected)

def test_cross_entropy_error_without_logistic():
  
  act = IdentityActivation()
  op = CrossEntropyLoss(act)
  x = rand_safe(10) #10 random numbers between 0 and 1
  y = rand_safe(10) #10 random numbers between 0 and 1

  # go for an exact match
  for p,q in zip(x,y):
    expected = (p-q)/(p*(1-p))
    assert is_close(op.error(p,q), expected), 'CrossEntropyLoss+IdentityActivation error does not perform as expected %g != %g' % (op.error(p,q), expected)

def test_cross_entropy_activation_detection():

  op = CrossEntropyLoss(LogisticActivation())
  assert op.logistic_activation

  op = CrossEntropyLoss(IdentityActivation())
  assert op.logistic_activation == False

def test_1d_ndarray():

  C = rand_safe0()
  op = SquareError(IdentityActivation())
  O = rand_safe(10) #10 random numbers between 0 and 1
  T = rand_safe(10) #10 random numbers between 0 and 1

  Y = op(O,T)
  assert Y.shape == O.shape
  assert Y.dtype == numpy.dtype(float)

  Y_f = op.f(O,T)
  assert Y_f.shape == O.shape
  assert Y.dtype == numpy.dtype(float)

  Y_f_prime = op.f_prime(O,T)
  assert Y_f_prime.shape == O.shape
  assert Y.dtype == numpy.dtype(float)

  Y_error = op.error(O,T)
  assert Y_error.shape == O.shape
  assert Y.dtype == numpy.dtype(float)

  for k,(o,t) in enumerate(zip(O,T)):
    assert is_close(op(o,t), Y[k])
    assert is_close(op.f(o,t), Y_f[k])
    assert is_close(op.f_prime(o,t), Y_f_prime[k])
    assert is_close(op.error(o,t), Y_error[k])

def test_2d_ndarray():

  C = rand_safe0()
  op = SquareError(IdentityActivation())
  O = rand_safe2(3,3)
  T = rand_safe2(3,3)

  Y = op(O,T)
  assert Y.shape == O.shape
  assert Y.dtype == numpy.dtype(float)

  Y_f = op.f(O,T)
  assert Y_f.shape == O.shape
  assert Y.dtype == numpy.dtype(float)

  Y_f_prime = op.f_prime(O,T)
  assert Y_f_prime.shape == O.shape
  assert Y.dtype == numpy.dtype(float)

  Y_error = op.error(O,T)
  assert Y_error.shape == O.shape
  assert Y.dtype == numpy.dtype(float)

  for k,(o,t) in enumerate(zip(O.flat,T.flat)):
    assert is_close(op(o,t), Y.flat[k])
    assert is_close(op.f(o,t), Y_f.flat[k])
    assert is_close(op.f_prime(o,t), Y_f_prime.flat[k])
    assert is_close(op.error(o,t), Y_error.flat[k])

def test_3d_ndarray():

  C = rand_safe0()
  op = SquareError(IdentityActivation())
  O = rand_safe3(3,3,3)
  T = rand_safe3(3,3,3)

  Y = op(O,T)
  assert Y.shape == O.shape
  assert Y.dtype == numpy.dtype(float)

  Y_f = op.f(O,T)
  assert Y_f.shape == O.shape
  assert Y.dtype == numpy.dtype(float)

  Y_f_prime = op.f_prime(O,T)
  assert Y_f_prime.shape == O.shape
  assert Y.dtype == numpy.dtype(float)

  Y_error = op.error(O,T)
  assert Y_error.shape == O.shape
  assert Y.dtype == numpy.dtype(float)

  for k,(o,t) in enumerate(zip(O.flat,T.flat)):
    assert is_close(op(o,t), Y.flat[k])
    assert is_close(op.f(o,t), Y_f.flat[k])
    assert is_close(op.f_prime(o,t), Y_f_prime.flat[k])
    assert is_close(op.error(o,t), Y_error.flat[k])

def test_4d_ndarray():

  C = rand_safe0()
  op = SquareError(IdentityActivation())
  O = rand_safe4(2,2,2,2)
  T = rand_safe4(2,2,2,2)

  Y = op(O,T)
  assert Y.shape == O.shape
  assert Y.dtype == numpy.dtype(float)

  Y_f = op.f(O,T)
  assert Y_f.shape == O.shape
  assert Y.dtype == numpy.dtype(float)

  Y_f_prime = op.f_prime(O,T)
  assert Y_f_prime.shape == O.shape
  assert Y.dtype == numpy.dtype(float)

  Y_error = op.error(O,T)
  assert Y_error.shape == O.shape
  assert Y.dtype == numpy.dtype(float)

  for k,(o,t) in enumerate(zip(O.flat,T.flat)):
    assert is_close(op(o,t), Y.flat[k])
    assert is_close(op.f(o,t), Y_f.flat[k])
    assert is_close(op.f_prime(o,t), Y_f_prime.flat[k])
    assert is_close(op.error(o,t), Y_error.flat[k])
