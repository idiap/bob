"""Torch is a toolkit of biometric utilities for face and speech processing.
"""

# This is a fix to the problem of using templates/static/exceptions/dynamic
# casts with boost.python. It makes all symbols loaded by python from this
# point onwards global
import sys, ctypes
default_flags = sys.getdlopenflags()
sys.setdlopenflags(default_flags|ctypes.RTLD_GLOBAL)

import core
import math
import config
import io
import measure
import sp
import ip
import db
import machine
import trainer

sys.setdlopenflags(default_flags)
del default_flags

__all__ = [
    'core', 
    'math',
    'config',
    'io',
    'measure',
    'sp',
    'ip',
    'db',
    'machine',
    'trainer'
    ]
