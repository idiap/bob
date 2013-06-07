#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Sun  2 Jun 16:42:52 2013 

"""Test activation functions
"""

import numpy
import math

from .. import IdentityActivation, \
    LinearActivation, \
    HyperbolicTangentActivation, \
    MultipliedHyperbolicTangentActivation, \
    LogisticActivation

from ...trainer import gradient

def test_identity():
  
  op = IdentityActivation()
  x = numpy.random.rand(10) #10 random numbers between 0 and 1

  # go for an exact match
  for k in x:
    assert op.f(k) == k, 'IdentityActivation does not perform identity %g != %g' % (op.f(k), k)

def test_identity_derivative():
  
  op = IdentityActivation()
  x = numpy.random.rand(10) #10 random numbers between 0 and 1

  # go for an exact match
  for k in x:
    assert op.f_prime(k) == 1., 'IdentityActivation derivative is not equal to 1.: %g != 1.' % (op.f_prime(k),)

  # tries to estimate the gradient and check
  for k in x:
    absdiff = abs(op.f_prime(k)-gradient.estimate(op.f,k))
    assert absdiff < 1e-4, 'IdentityActivation derivative and estimation do not match to 10^-4: |%g-%g| = %g' % (op.f_prime(k), gradient.estimate(op.f,k), absdiff)

def test_linear():
 
  C = numpy.random.rand()
  op = LinearActivation(C)
  x = numpy.random.rand(10) #10 random numbers between 0 and 1

  # go for an exact match
  for k in x:
    assert op.f(k) == (C*k), 'LinearActivation does not match expected value: %g != %g' % (op.f(k), C*k)

def test_linear_derivative():
 
  C = numpy.random.rand()
  op = LinearActivation(C)
  x = numpy.random.rand(10) #10 random numbers between 0 and 1

  # go for an exact match
  for k in x:
    assert op.f_prime(k) == C, 'LinearActivation derivative does not match expected value: %g != %g' % (op.f_prime(k), k)
  
  # tries to estimate the gradient and check
  for k in x:
    absdiff = abs(op.f_prime(k)-gradient.estimate(op.f,k))
    assert absdiff < 1e-4, 'IdentityActivation derivative and estimation do not match to 10^-4: |%g-%g| = %g' % (op.f_prime(k), gradient.estimate(op.f,k), absdiff)

def test_hyperbolic_tangent():
 
  op = HyperbolicTangentActivation()
  x = numpy.random.rand(10) #10 random numbers between 0 and 1

  # go for an exact match
  for k in x:
    assert op.f(k) == math.tanh(k), 'HyperbolicTangentActivation does not match expected value: %g != %g' % (op.f(k), math.tanh(k))

def test_hyperbolic_tangent_derivative():
 
  op = HyperbolicTangentActivation()
  x = numpy.random.rand(10) #10 random numbers between 0 and 1

  # go for an exact match
  for k in x:
    precise = 1 - op.f(k)**2
    assert op.f_prime(k) == precise, 'HyperbolicTangentActivation derivative does not match expected value: %g != %g' % (op.f_prime(k), precise)
  
  # tries to estimate the gradient and check
  for k in x:
    absdiff = abs(op.f_prime(k)-gradient.estimate(op.f,k))
    assert absdiff < 1e-4, 'HyperbolicTangentActivation derivative and estimation do not match to 10^-4: |%g-%g| = %g' % (op.f_prime(k), gradient.estimate(op.f,k), absdiff)

def test_logistic():
 
  op = LogisticActivation()
  x = numpy.random.rand(10) #10 random numbers between 0 and 1

  # go for an exact match
  for k in x:
    precise = 1. / (1. + math.exp(-k))
    assert op.f(k) == precise, 'LogisticActivation does not match expected value: %g != %g' % (op.f(k), precise)

def test_logistic_derivative():
 
  op = LogisticActivation()
  x = numpy.random.rand(10) #10 random numbers between 0 and 1

  # go for an exact match
  for k in x:
    precise = op.f(k) * (1 - op.f(k))
    assert op.f_prime(k) == precise, 'LogisticActivation derivative does not match expected value: %g != %g' % (op.f_prime(k), precise)
  
  # tries to estimate the gradient and check
  for k in x:
    absdiff = abs(op.f_prime(k)-gradient.estimate(op.f,k))
    assert absdiff < 1e-4, 'LogisticActivation derivative and estimation do not match to 10^-4: |%g-%g| = %g' % (op.f_prime(k), gradient.estimate(op.f,k), absdiff)

def test_multiplied_tanh():

  C = numpy.random.rand()
  M = numpy.random.rand()
  op = MultipliedHyperbolicTangentActivation(C, M)
  x = numpy.random.rand(10) #10 random numbers between 0 and 1

  # go for an exact match
  for k in x:
    assert op.f(k) == C*math.tanh(M*k), 'MultipliedHyperbolicTangentActivation does not match expected value: %g != %g' % (op.f(k), C*math.tanh(M*k))

def test_multiplied_tanh_derivative():
 
  C = numpy.random.rand()
  M = numpy.random.rand()
  op = MultipliedHyperbolicTangentActivation(C, M)
  x = numpy.random.rand(10) #10 random numbers between 0 and 1

  # go for an exact match
  for k in x:
    precise = C*M*(1-math.pow(math.tanh(M*k),2))
    assert op.f_prime(k) == precise, 'MultipliedHyperbolicTangentActivation derivative does not match expected value: %g != %g' % (op.f_prime(k), precise)
 
  # tries to estimate the gradient and check
  for k in x:
    absdiff = abs(op.f_prime(k)-gradient.estimate(op.f,k))
    assert absdiff < 1e-4, 'MultipliedHyperbolicTangentActivation derivative and estimation do not match to 10^-4: |%g-%g| = %g' % (op.f_prime(k), gradient.estimate(op.f,k), absdiff)
