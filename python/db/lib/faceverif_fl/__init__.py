#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>
# Wed 28 May 06:37:33 2011 

"""A face verification database based on file lists
"""

# Use this variable to tell dbmanage.py all driver that there is nothing to
# download for this database.
__builtin__ = True

def dbname():
  """Calculates my own name automatically."""
  import os
  return os.path.basename(os.path.dirname(__file__))

from .query import Database
from .commands import add_commands
