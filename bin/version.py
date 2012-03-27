#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 27 Mar 2012 08:42:30 CEST 

"""Calculates the version number based on some indicators.

By default, the version is extracted from the package directory name. If that
is not available, use the git command to extract the current hash.
"""

import os
import re
import sys
import subprocess

def git_hash():
  """Tries to find if the are any matching tags to this release"""

  try:
    p = subprocess.Popen(['git', 'log', "--pretty=format:%h", 
      '--date=short', '-n', '1'], stdout=subprocess.PIPE, stdin=None)
    stdout, stderr = p.communicate()
    return stdout
  except:
    pass

  return None

def git_pretty_hash():
  """Tries to find if the are any matching tags to this release"""

  try:
    p = subprocess.Popen(['git', 'log', "--pretty=format:git-%cd-%h",
      '--date=short', '-n', '1'], stdout=subprocess.PIPE, stdin=None)
    stdout, stderr = p.communicate()
    return stdout
  except:
    pass

  return None

class null(object):
  """A look-alike stream that discards the input"""

  def write(self, s):
    pass
  
  def flush(self):
    pass

def git_tag():
  """See if there are any git tags that describe the current checkout"""

  try:
    h = git_hash()
    if not h: return None
    p = subprocess.Popen(['git', 'describe', '--tags', h],
        stdout=subprocess.PIPE, stderr=null(), stdin=None)
    stdout, stderr = p.communicate()
    return stdout
  except:
    pass

  return None

def package_version():
  """Interprets the package version from the directory name"""

  dirname = os.path.basename(os.path.realpath(os.curdir))

  match = re.match(r'^bob[-_](.*)$', dirname)

  if match: return match.groups()[0]

  return None

if __name__ == '__main__':

  # change directories to my parent's
  os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0]))))

  version = package_version()

  if not version:
    # the second thing to try is git - is there any tag associated?
    version = git_tag()

  if not version:
    # the third thing to try is git again, get the current abbreviated hash
    version = git_pretty_hash()

  if not version:
    version = 'unknown'

  print version

  sys.exit(0)
