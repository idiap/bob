#!/usr/bin/env python
# vim: set fileencoding=utf-8 : 
# Andre Anjos <andre.anjos@idiap.ch> 
# Sat 23 Feb 06:00:13 2013 CET

"""Base tools for dealing with arrays, random numbers and our Python/C++ bridge
"""

import sys
import ctypes
import logging

def __resolve_name__(name, package, level):
  """Return the absolute name of the module to be imported."""

  if not hasattr(package, 'rindex'):
    raise ValueError("'package' not set to a string")
  dot = len(package)
  for x in xrange(level, 1, -1):
    try:
      dot = package.rindex('.', 0, dot)
    except ValueError:
      raise ValueError("attempted relative import beyond top-level package")
  return "%s.%s" % (package[:dot], name)

def __import_extension__(name, package, locals):
  """Import a module.

  The 'package' argument is required when performing a relative import. It
  specifies the package to use as the anchor point from which to resolve the
  relative import to an absolute import.

  """

  # This is a fix to the problem of using templates/static/exceptions/dynamic
  # casts with boost.python. It makes all symbols loaded by python from this
  # point onwards global
  default_flags = sys.getdlopenflags()
  sys.setdlopenflags(default_flags|ctypes.RTLD_GLOBAL)

  if name.startswith('.'):
    if not package:
      raise TypeError("relative imports require the 'package' argument")
    level = 0
    for character in name:
      if character != '.':
        break
      level += 1
    name = __resolve_name__(name[level:], package, level)
  __import__(name, locals=locals)
  sys.setdlopenflags(default_flags)

  return sys.modules[name]

def __map_extension__(module, locals):
  """Map all symbols from the extension locally."""

  names = getattr(module, '__all__', None)
  if names is None:
    names = [name for name in dir(module) if not name.startswith('_')]
  for name in names:
    locals[name] = getattr(module, name)

def __from_extension_import__(name, package, locals, fromlist=['*']):
  """Imports all objects from the extension"""

  module = __import_extension__(name, package, locals)

  if '*' in fromlist:
    __map_extension__(module, locals)
  else:
    for name in fromlist: locals[name] = getattr(module, name)

__from_extension_import__('._core', __package__, locals())
from . import array
from . import random
__all__ = [k for k in dir() if not k.startswith('_')]

# get the default logger of Bob
logger = logging.getLogger('bob')

# by default, warning and error messages should be written to sys.stderr
warn_err = logging.StreamHandler(sys.stderr)
warn_err.setLevel(logging.WARNING)
logger.addHandler(warn_err)

# debug and info messages are written to sys.stdout
class InfoFilter:
  def filter(self, record): return record.levelno <= logging.INFO
debug_info = logging.StreamHandler(sys.stdout)
debug_info.setLevel(logging.DEBUG)
debug_info.addFilter(InfoFilter())
logger.addHandler(debug_info)

# this will setup divergence from C++ into python.logging correctly
cxx_logger = logging.getLogger('bob.c++')
debug.reset(PythonLoggingOutputDevice(cxx_logger.debug))
info.reset(PythonLoggingOutputDevice(cxx_logger.info))
warn.reset(PythonLoggingOutputDevice(cxx_logger.warn))
error.reset(PythonLoggingOutputDevice(cxx_logger.error))
