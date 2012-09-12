from ._core import *
from . import array
from . import random
__all__ = dir()

#this will setup divergence from C++ into python.logging correctly
import logging

#this configures our core logger
logger = logging.getLogger("bob")
ch = logging.StreamHandler()
formatter = logging.Formatter("%(name)s@%(asctime)s|%(levelname)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

cxx_logger = logging.getLogger('bob.c++')
cxx_logger.setLevel(logging.WARNING)
debug.reset(PythonLoggingOutputDevice(cxx_logger.debug))
info.reset(PythonLoggingOutputDevice(cxx_logger.info))
warn.reset(PythonLoggingOutputDevice(cxx_logger.warn))
error.reset(PythonLoggingOutputDevice(cxx_logger.error))
