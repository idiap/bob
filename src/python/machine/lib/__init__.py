from libpytorch_machine import *

def linearmachine_repr(self):
  """A funky way to display a torch Linear Machine"""
  if self.activation == Activation.LINEAR:
    return '<LinearMachine %s@%s>' % (self.weights.cxx_element_typename, self.weights.shape())
  else:
    return '<LinearMachine %s@%s [act: %s]>' % (self.weights.cxx_element_typename, self.weights.shape(), self.activation)
LinearMachine.__repr__ = linearmachine_repr
del linearmachine_repr

def linearmachine_str(self):
  """A funky way to print a torch Linear Machine"""
  act = ""
  if self.activation != Activation.LINEAR:
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
  shape = self.weights.shape()
  return 'LinearMachine (%s) %d inputs, %d outputs%s%s%s%s\n %s' % \
      (self.weights.cxx_element_typename, shape[0], shape[1], act, sub, div,
          bias, self.weights)
LinearMachine.__str__ = linearmachine_str
del linearmachine_str
