"""A signal processing and machine learning toolkit for biometrics.
"""

# This is a fix to the problem of using templates/static/exceptions/dynamic
# casts with boost.python. It makes all symbols loaded by python from this
# point onwards global
import sys, ctypes
default_flags = sys.getdlopenflags()
sys.setdlopenflags(default_flags|ctypes.RTLD_GLOBAL)

import core
import io
import math
import sp
import ip
import measure
import db
import machine
import trainer

try:
  # the visioner may not be built if Qt4 is not installed
  import visioner
  has_visioner = True
except ImportError:
  has_visioner = False

sys.setdlopenflags(default_flags)
del default_flags

__all__ = [
    'core', 
    'math',
    'io',
    'measure',
    'sp',
    'ip',
    'db',
    'machine',
    'trainer'
    ]

if has_visioner: __all__.append('visioner')
