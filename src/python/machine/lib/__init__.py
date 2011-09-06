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

def mlp_repr(self):
  """A funky way to display a torch MLP"""
  bias = False
  for i, k in enumerate(self.biases): 
    if not (k == 0.0).all(): bias = True
  return '<MLP %s@%s [bias: %s][act: %s]>' % (self.weights[0].cxx_element_typename, self.shape, str(bias).lower(), self.activation)
MLP.__repr__ = mlp_repr
del mlp_repr

def mlp_str(self):
  """A funky way to print a torch MLP"""
  act = "[act: %s]" % self.activation
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
      (self.weights[0].cxx_element_typename, self.shape, str(has_bias).lower(), 
          act, sub, div, bias, weight)
MLP.__str__ = mlp_str
del mlp_str

def ztnormSameValue(vect_A, vect_B):
  """Computes the matrix of boolean D for the ZT-norm, which indicates where 
     the client ids of the T-Norm models and Z-Norm samples match.

     vect_A An (ordered) list of client_id corresponding to the T-Norm models
     vect_B An (ordered) list of client_id corresponding to the Z-Norm impostor samples
  """
  from .. import core
  sameMatrix = core.array.bool_2(len(vect_A), len(vect_B))
  for j in range(len(vect_A)):
    for i in range(len(vect_B)):
      sameMatrix[j, i] = (vect_A[j] == vect_B[i])
  return sameMatrix
