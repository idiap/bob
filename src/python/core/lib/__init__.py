from libpytorch_core import *
import libpytorch_core_array as array
from . import mapstring
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

def version_table():
  """Returns a summarized version table of all software we depend on."""
  retval = {} 

  retval['compiler'] = '-'.join(compiler_version())
  retval['blitz++'] = blitz_version()
  retval['boost'] = boost_version()
  retval['python'] = python_version()
  retval['hdf5'] = hdf5_version()
  retval['numpy'] = numpy_version()
  retval['ffmpeg'] = ';'.join(['-'.join(k) for k in ffmpeg_version()])
  retval['image magick'] = magick_version()
  retval['matio'] = matio_version()

  return retval

def version_string():
  """Returns a string representation of the return value of version_table()"""

  retval = [] 
  
  table = version_table()
  for key in sorted(table.keys()):
    retval.append(key.capitalize() + ': ' + table[key])

  return '\n'.join(retval)
