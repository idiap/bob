#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>

"""Dumps lists of files.
"""

import os
import sys

def dumplist(args):
  from .query import Database
  db = Database()

  r = db.files(directory=args.directory, extension=args.extension,
      protocol=args.protocol, purposes=args.purposes, client_ids=args.client_ids, 
      groups=args.groups, languages=args.languages)
  for id, f in r.items(): 
    print '%s' % (f)
