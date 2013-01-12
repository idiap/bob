"""A signal processing and machine learning toolkit for biometrics.
"""

# This is a fix to the problem of using templates/static/exceptions/dynamic
# casts with boost.python. It makes all symbols loaded by python from this
# point onwards global
import sys, ctypes
default_flags = sys.getdlopenflags()
sys.setdlopenflags(default_flags|ctypes.RTLD_GLOBAL)

from . import core
from . import io
from . import math
from . import measure
from . import sp
from . import ip
from . import ap
from . import db
from . import machine
from . import trainer
  
try:
  # the daq may not be built if Qt4 is not installed
  from . import visioner
  has_visioner = True
except ImportError:
  has_visioner = False

try:
  # the daq may not be built if Qt4 is not installed
  from . import daq
  has_daq = True
except ImportError:
  has_daq = False


sys.setdlopenflags(default_flags)
del default_flags

__all__ = [
    'core', 
    'math',
    'io',
    'measure',
    'sp',
    'ip',
    'ap',
    'db',
    'machine',
    'trainer',
    'build',
    'helper',
    ]

if has_visioner: __all__.append('visioner')
if has_daq: __all__.append('daq')
version = __import__('pkg_resources').require('bob')[0].version
