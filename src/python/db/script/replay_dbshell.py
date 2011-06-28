#!/idiap/group/torch5spro/nightlies/last/bin/shell.py -- python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Thu 12 May 13:06:05 2011 

"""Runs a DB interactive shell for the user
"""

import sys
from replay import dbshell
sys.exit(dbshell())
