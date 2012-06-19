#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Salim Kayal <salim.kayal@idiap.ch>

"""The FIR database
"""

def dbname():
  """Calculates my own name automatically."""
  import os
  return os.path.basename(os.path.dirname(__file__))

from .query import Database
from .commands import add_commands
