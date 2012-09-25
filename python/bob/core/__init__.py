from ._core import *
from . import array
from . import random
__all__ = dir()

import logging
import sys

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
