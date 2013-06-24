#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Mon 24 Jun 11:01:07 2013 

"""MLP additions
"""

from . import MLP

def mlp_repr(self):
  """A funky way to display a bob MLP"""
  bias = False
  for i, k in enumerate(self.biases): 
    if not (k == 0.0).all(): bias = True
  if self.hidden_activation != self.output_activation:
    return '<MLP %s@%s [bias: %s][act: {hidden}%s/{output}%s]>' % (self.weights[0].dtype, self.shape, str(bias).lower(), self.hidden_activation, self.output_activation)
  else:
    return '<MLP %s@%s [bias: %s][act: %s]>' % (self.weights[0].dtype, self.shape, str(bias).lower(), self.hidden_activation)
MLP.__repr__ = mlp_repr
del mlp_repr

def mlp_str(self):
  """A funky way to print a bob MLP"""
  if self.hidden_activation != self.output_activation:
    act = "[act: {hidden}%s/{output}%s]" % (self.hidden_activation, self.output_activation)
  else:
    act = "[act: %s]" % self.hidden_activation
  sub = ""
  if not (self.input_subtract == 0.0).all():
    sub = "\n subtract: %s" % self.input_subtract
  div = ""
  if not (self.input_divide == 1.0).all():
    div = "\n divide: %s" % self.input_divide
  has_bias = False
  bias = ""
  for i, k in enumerate(self.biases):
    if not (k == 0.0).all():
      has_bias = True
      bias += "\n bias[%d]:\n %s" % (i, k)
  weight = ""
  for i, k in enumerate(self.weights):
    weight += "\n weight[%d]:\n %s" % (i, k)
  return 'MLP %s@%s [bias: %s]%s%s%s%s%s' % \
      (self.weights[0].dtype, self.shape, str(has_bias).lower(), 
          act, sub, div, bias, weight)
MLP.__str__ = mlp_str
del mlp_str
