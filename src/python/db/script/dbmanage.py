#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 28 Jun 2011 13:55:35 CEST 

"""This script drives all commands from the specific database subdrivers.
"""

epilog = """  For a list of available databases:
  >>> %(prog)s --help

  For a list of actions on a database:
  >>> %(prog)s <database-name> --help
"""

from bob.db.manage import *

if __name__ == '__main__':

  from argparse import RawDescriptionHelpFormatter
  parser = create_parser(description=__doc__, epilog=epilog,
      formatter_class=RawDescriptionHelpFormatter) 
  args = parser.parse_args()
  args.func(args)
