"""A signal processing and machine learning toolkit for biometrics.
"""

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

version = __import__('pkg_resources').require('bob')[0].version

__all__ = [k for k in dir() if not k.startswith('_')]
