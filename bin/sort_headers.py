#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue  4 Sep 21:43:59 2012 

"""Sort headers in the following order:

  1. CMake prefix paths
  2. Other paths (paths that do not start with the compiler location or
     satisfy item 1)
  3. Paths that are on the compiler location (do not satisfy 1 or 2)

  The program is passed 3 parameters:

  1. The location of c++ compiler
  2. The value of the CMAKE_PREFIX_PATH variable
  3. The list of headers directories to parse
"""

import os
import sys

def uniq(seq, idfun=None): 
  # order preserving
  if idfun is None:
    def idfun(x): return x
  seen = {}
  result = []
  for item in seq:
    marker = idfun(item)
    # in old Python versions:
    # if seen.has_key(marker)
    # but in new ones:
    if marker in seen: continue
    seen[marker] = 1
    result.append(item)
  return result

cxx = sys.argv[1]
cxx_basepath = os.path.dirname(os.path.dirname(os.path.abspath(cxx)))
prefixes = sys.argv[2].split(os.pathsep)
dirs = sys.argv[3].split(';')

def share_root(base, directory):
  return os.path.commonprefix((base, directory)) == base

path_group_1 = []
if prefixes:
  for prefix in prefixes:
    path_group_1 += [k for k in dirs if share_root(prefix, k)]
  path_group_1 = uniq(path_group_1)

  # Remove group 1 directories from input list
  dirs = [k for k in dirs if k not in path_group_1]
#print "Group 1 (on prefixes):", path_group_1

path_group_2 = uniq([k for k in dirs if not share_root(cxx_basepath, k)])
#print "Group 1 (not on compiler):", path_group_2

# Remove group 2 directories from input list
path_group_3 = uniq([k for k in dirs if k not in path_group_2])
#print "Group 3 (everything else):", path_group_3

print ';'.join(path_group_1 + path_group_2 + path_group_3)
