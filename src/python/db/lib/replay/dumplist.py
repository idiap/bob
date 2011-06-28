#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Thu 12 May 14:02:28 2011 

"""Dumps lists of files.
"""

import os
import sys

def dumplist(args):
  from .query import Database
  db = Database()

  r = db.files(directory=args.directory, extension=args.extension,
      device=args.device, support=args.support, groups=args.group, cls=args.cls)
  for id, f in r.items(): 
    print '%s' % (f)
