#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Wed 18 May 09:28:44 2011 

"""The Idiap Replay attack database consists of Photo and Video attacks to
different identities under different illumination conditions.
"""

def dbname():
  """Calculates my own name automatically."""
  import os
  return os.path.basename(os.path.dirname(__file__))

from .utils import location, dbshell
from .query import Database
from .commands import add_commands
