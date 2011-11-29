from libpytorch_core import *
from . import array
from . import vector
from . import boost_tuple
from . import random
__all__ = dir()

#this will setup divergence from C++ into python.logging correctly
import logging

#this configures our core logger
logger = logging.getLogger("torch")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(name)s@%(asctime)s|%(levelname)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

cxx_logger = logging.getLogger('torch.cxx')
debug.reset(PythonLoggingOutputDevice(cxx_logger.debug))
info.reset(PythonLoggingOutputDevice(cxx_logger.info))
warn.reset(PythonLoggingOutputDevice(cxx_logger.warn))
error.reset(PythonLoggingOutputDevice(cxx_logger.error))

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
