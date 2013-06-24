#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Mon 24 Jun 11:03:59 2013 

"""LinearMachine additions
"""

from . import LinearMachine, IdentityActivation

def linearmachine_repr(self):
  """A funky way to display a bob Linear Machine"""
  if self.activation == IdentityActivation():
    return '<LinearMachine %s@%s>' % (self.weights.dtype, self.weights.shape)
  else:
    return '<LinearMachine %s@%s [act: %s]>' % (self.weights.dtype, self.weights.shape, self.activation)
LinearMachine.__repr__ = linearmachine_repr
del linearmachine_repr

def linearmachine_str(self):
  """A funky way to print a bob Linear Machine"""
  act = ""
  if self.activation != IdentityActivation():
    act = " [act: %s]" % self.activation
  sub = ""
  if not (self.input_subtract == 0.0).all():
    sub = "\n subtract: %s" % self.input_subtract
  div = ""
  if not (self.input_divide == 1.0).all():
    div = "\n divide: %s" % self.input_divide
  bias = ""
  if not (self.biases == 0.0).all():
    bias = "\n bias: %s" % self.biases
  shape = self.weights.shape
  return 'LinearMachine (%s) %d inputs, %d outputs%s%s%s%s\n %s' % \
      (self.weights.dtype, shape[0], shape[1], act, sub, div,
          bias, self.weights)
LinearMachine.__str__ = linearmachine_str
del linearmachine_str
