from libpytorch_core import *
__all__ = dir()

def variables(self):
  """Returns all available variables in a list."""
  retval = []
  for k in range(self.nVariables()): retval.append(self.variable(k))
  return retval

def variable_dict(self):
  """Returns all variables in a dictionary."""
  retval = {} 
  for k in range(self.nVariables()): 
    v = self.variable(k)
    retval[v.name] = v
  return retval

Object.variables = variables
Object.variable_dict = variable_dict

def tensor_str(i):
  v = 'Tensor(type=%d, dimensions=%d)[' % (i.getDatatype(), i.nDimension())
  dimensions = []
  for k in range(i.nDimension()): dimensions.append(str(i.size(k)))
  return v + ','.join(dimensions) + ']'
Tensor.__str__ = tensor_str
