from ..core import __from_extension_import__
__from_extension_import__('._machine', __package__, locals())

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

def ztnorm_same_value(vect_a, vect_b):
  """Computes the matrix of boolean D for the ZT-norm, which indicates where 
     the client ids of the T-Norm models and Z-Norm samples match.

     vect_a An (ordered) list of client_id corresponding to the T-Norm models
     vect_b An (ordered) list of client_id corresponding to the Z-Norm impostor samples
  """
  from .. import core
  import numpy as np
  sameMatrix = np.ndarray((len(vect_a), len(vect_b)), 'bool')
  for j in range(len(vect_a)):
    for i in range(len(vect_b)):
      sameMatrix[j, i] = (vect_a[j] == vect_b[i])
  return sameMatrix

__all__ = [k for k in dir() if not k.startswith('_')]
