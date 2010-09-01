"""Torch is a toolkit of biometric utilities for face and speech processing.
"""

# This is a fix to the problem of using templates/static/exceptions/dynamic
# casts with boost.python. It makes all symbols loaded by python from this
# point onwards global
import sys, dl
default_flags = sys.getdlopenflags()
sys.setdlopenflags(dl.RTLD_NOW | dl.RTLD_GLOBAL)

import core
import sp
import ip
import scanning

sys.setdlopenflags(default_flags)
del default_flags, sys, dl

__all__ = ['core', 'sp', 'ip', 'scanning']
