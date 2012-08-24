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
from distutils.version import StrictVersion

# Set this variable to the value of the last known stable release +1. The
# number to +1 will depend on which branch you are in. If you are in the master
# branch, you will have to +1 either the first or the second digit. If you are
# on a specific stable branch (e.g. 1.0), you will have to +1 the last digit.
# The objective of this variable is to produce release numbers that are
# consistent and growing with time. This is used by setuptools to determine
# which distribution of bob to use. Be aware.
NEXT_VERSION = '1.1.0'

BRANCH_RE = re.compile(r'^\d+\.\d+$')
TAG_RE = re.compile(r'^v\d+\.\d+\.\d+$')
VERSION_RE = re.compile(r'^\d+\.\d+\.\d+$')

def git_version_branches():
  """Get the branches available on the origin"""
  
  try:
    p = subprocess.Popen(['git', 'branch', '-a'], stdout=subprocess.PIPE, 
        stdin=None)
    stdout, stderr = p.communicate()
    
    if p.returncode != 0: raise RuntimeError
   
    cand = list(set([j.strip('* ').split('/')[-1] for j in stdout.split('\n')]))
    return [k for k in cand if k and BRANCH_RE.match(k)]

  except:
    print "Warning: could retrieve branch list"

def git_current_branch():
  """Get the current branch we are sitting on"""
  
  try:
    p = subprocess.Popen(['git', 'branch'], stdout=subprocess.PIPE, stdin=None)
    stdout, stderr = p.communicate()

    if p.returncode != 0: raise RuntimeError
    
    for k in stdout.split('\n'):
      if not k: continue
      if k[0] == '*':
        return k[2:].strip()
 
    # if you get to this, point something went wrong
    raise RuntimeError

  except:
    print "Warning: could not determine in which branch I'm on"

def git_next_minor_version(branch):
  """Gets the next minor version"""

  try:
    p = subprocess.Popen(['git', 'tag', '-l'], stdout=subprocess.PIPE, 
        stdin=None)
    stdout, stderr = p.communicate()

    if p.returncode != 0: raise RuntimeError

    candidates = [k.strip() for k in stdout.split('\n')]
    candidates = sorted([StrictVersion(k[1:]) for k in candidates if k and TAG_RE.match(k)])

    if not candidates: raise RuntimeError

    next_version = list(candidates[-1].version)
    next_version[2] += 1

    return '.'.join([str(k) for k in next_version])
 
  except:
    print "Warning: could not determine latest tag on branch (%s). Assuming it is %s.0" % (branch, branch)
    return branch + '.0'

def git_next_major_version():
  """Gets the next major version"""

  last = sorted([StrictVersion(k) for k in git_version_branches()])[-1]

  next_version = list(last.version)[0:2]
  next_version[1] += 1
  next_version.append(0)

  return '.'.join([str(k) for k in next_version])

def git_next_version(branch):
  """Gets the next version given the branch I'm on"""

  if BRANCH_RE.match(branch):
    # we are on a stable branch
    return git_next_minor_version(branch)

  elif branch == 'master':
    # we are on the master tip
    return git_next_major_version()

  else:
    print "Warning: not on 'master' or any known stable branch. Cannot guess next version"

def git_count(branch):
  """Count the number of commits in the repository.
  
  Note: This does not work right for shallow git clones.
  """

  try:
    p = subprocess.Popen(['git', 'rev-list', branch],
        stdout=subprocess.PIPE, stdin=None)
    stdout, stderr = p.communicate()
    
    if p.returncode != 0: raise RuntimeError
    
    return stdout.count('\n')

  except:
    print "Warning: could not determine commit count on branch '%s'" % branch

def package_version():
  """Interprets the package version from the directory name"""

  dirname = os.path.basename(os.path.realpath(os.curdir))

  match = re.match(r'^bob[-_](\d+\.\d+\.\d+[abcd]\d+)$', dirname)

  if match: return match.groups()[0]

  return None

if __name__ == '__main__':

  import optparse

  parser = optparse.OptionParser()
  parser.add_option("-v", "--version", dest="version", metavar='VERSION', help="force version to a given string (format M.m.p; by default calculates the version number of the next build based on the currently existing branch and tags)", type='str')
  parser.add_option("-l", "--letter", dest="letter", metavar='LETTER', choices=('a', 'b', 'c'), default='a', help="force the suffix letter to be one of ([a]alpha, [b]beta, [c]candidate; defaults to 'a')")
  parser.add_option("-c", "--counter", dest="count", metavar='COUNT', type='int', help="force the counter after the letter (by default use number of commits in the current branch)")

  (options, args) = parser.parse_args()

  if options.version:
    if not VERSION_RE.match(options.version):
      parser.error("input version has to conform to the format M.m.p (where M, m and p are non-negative integers")

  if options.count is not None and options.count < 0:
    parser.error("count has to be greater or equal zero")

  if args:
    parser.error("this program does not accept positional parameters")

  # change directories to my parent's
  os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0]))))

  branch = git_current_branch()

  if options.version is None:
    options.version = package_version()

    if not options.version:
      if branch is not None:
        options.version =  git_next_version(branch)
        
  if options.count is None:
    options.count = git_count(branch)

  if not options.version: print 'unknown'
  else: 
    final = options.version + options.letter + str(options.count)
    StrictVersion(final) #double-checks all is good
    print final

  sys.exit(0)
