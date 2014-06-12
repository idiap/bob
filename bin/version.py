#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 27 Mar 2012 08:42:30 CEST
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Calculates the version number based on some indicators.

By default, the version is extracted from the package directory name. If that
is not available, use the git command to extract the current hash.
"""

import os
import re
import sys
import subprocess
from distutils.version import StrictVersion

BRANCH_RE = re.compile(r'^\d+\.\d+$')
TAG_RE = re.compile(r'^v\d+\.\d+\.\d+([abc]\d+)?$')
VERSION_RE = re.compile(r'^\d+\.\d+\.\d+([abc]\d+)?$')
BOB_REPOSITORY = 'https://github.com/idiap/bob.git'

def git_remote_version_branches(verbose):
  """Get the branches available on the origin using git ls-remote"""

  try:
    p = subprocess.Popen(['git', 'ls-remote', '--heads', BOB_REPOSITORY],
        stdout=subprocess.PIPE, stdin=None)
    stdout, stderr = p.communicate()

    if p.returncode != 0: raise RuntimeError

    # Python3 support
    if isinstance(stdout, bytes) and not isinstance(stdout, str):
      stdout = stdout.decode('utf8')

    cand = [k[-1].split('/')[-1] for k in [j.split() for j in stdout.split('\n')] if k]
    return [k for k in cand if BRANCH_RE.match(k)]

  except:
    if verbose:
      print("Warning: could retrieve remote branch list")

def git_version_branches(verbose):
  """Get the branches available on the origin"""

  try:
    p = subprocess.Popen(['git', 'branch', '-a'], stdout=subprocess.PIPE,
        stdin=None)
    stdout, stderr = p.communicate()

    if p.returncode != 0: raise RuntimeError

    # Python3 support
    if isinstance(stdout, bytes) and not isinstance(stdout, str):
      stdout = stdout.decode('utf8')

    cand = list(set([j.strip('* ').split('/')[-1] for j in stdout.split('\n')]))
    return [k for k in cand if k and BRANCH_RE.match(k)]

  except:
    if verbose:
      print("Warning: could retrieve branch list")

def git_current_branch(verbose):
  """Get the current branch we are sitting on"""

  try:
    p = subprocess.Popen(['git', 'branch'], stdout=subprocess.PIPE, stdin=None)
    stdout, stderr = p.communicate()

    if p.returncode != 0: raise RuntimeError

    # Python3 support
    if isinstance(stdout, bytes) and not isinstance(stdout, str):
      stdout = stdout.decode('utf8')

    for k in stdout.split('\n'):
      if not k: continue
      if k[0] == '*':
        return k[2:].strip()

    # if you get to this, point something went wrong
    raise RuntimeError

  except:
    if verbose:
      print("Warning: could not determine in which branch I'm on")

def git_next_minor_version(branch, verbose):
  """Gets the next minor version"""

  try:
    p = subprocess.Popen(['git', 'tag', '-l'], stdout=subprocess.PIPE,
        stdin=None)
    stdout, stderr = p.communicate()

    if p.returncode != 0: raise RuntimeError

    # Python3 support
    if isinstance(stdout, bytes) and not isinstance(stdout, str):
      stdout = stdout.decode('utf8')

    candidates = [k.strip() for k in stdout.split('\n')]
    candidates = sorted([StrictVersion(k[1:]) for k in candidates if k and TAG_RE.match(k)])

    if not candidates: raise RuntimeError

    candidate = candidates[-1]
    if candidate.prerelease:
      next_version = list(candidate.version)
      return '.'.join([str(k) for k in next_version]) + \
              candidate.prerelease[0] + str(candidate.prerelease[1] + 1)
    else:
      next_version = list(candidate.version)
      next_version[2] += 1
      return '.'.join([str(k) for k in next_version])

  except:
    if verbose:
      print("Warning: could not determine latest tag on branch (%s). Assuming it is %s.0" % (branch, branch))
    return branch + '.0'

def git_next_major_version(verbose):
  """Gets the next major version"""

  env = 'BOB_VERSION_ONLY_REMOTE'
  candidates = None
  if env not in os.environ or (env in os.environ and os.environ[env].lower() in ('', '0', 'false', 'off', 'no')):
    # try local
    candidates = sorted([StrictVersion(k) for k in git_version_branches(verbose)])

  if not candidates:
    # try remote
    candidates = \
        sorted([StrictVersion(k) for k in git_remote_version_branches(verbose)])

  if not candidates:
    return None

  last = candidates[-1]

  next_version = list(last.version)[0:2]
  next_version[1] += 1
  next_version.append(0)

  # special cases
  if next_version == [1, 3, 0]: next_version = [2, 1, 0]

  return '.'.join([str(k) for k in next_version])

def git_next_version(branch, verbose):
  """Gets the next version given the branch I'm on"""

  if BRANCH_RE.match(branch):
    # we are on a stable branch
    return git_next_minor_version(branch, verbose)

  else:
    # we are on the master tip
    return git_next_major_version(verbose)

def git_count(branch, verbose):
  """Count the number of commits in the repository.

  Note: This does not work right for shallow git clones.
  """

  try:
    p = subprocess.Popen(['git', 'rev-list', branch],
        stdout=subprocess.PIPE, stdin=None)
    stdout, stderr = p.communicate()

    if p.returncode != 0: raise RuntimeError

    # Python3 support
    if isinstance(stdout, bytes) and not isinstance(stdout, str):
      stdout = stdout.decode('utf8')

    return stdout.count('\n')

  except:
    if verbose:
      print("Warning: could not determine commit count on branch '%s'" % branch)

def package_version(verbose):
  """Interprets the package version from the directory name"""

  dirname = os.path.basename(os.path.realpath(os.curdir))

  match = re.match(r'^bob[-_](\d+\.\d+\.\d+([abcd]\d+)?)$', dirname)

  if match: return match.groups()[0]

  return None

if __name__ == '__main__':

  import optparse

  parser = optparse.OptionParser()
  parser.add_option("-v", "--version", dest="version", metavar='VERSION', help="force version to a given string (format M.m.p; by default calculates the version number of the next build based on the currently existing branch and tags)", type='str')
  parser.add_option("-l", "--letter", dest="letter", metavar='LETTER', choices=('a', 'b', 'c'), help="force the suffix letter to be one of ([a]alpha, [b]beta, [c]candidate; defaults to None)")
  parser.add_option("-c", "--counter", dest="count", metavar='COUNT', type='int', help="force the counter after the letter (by default use number of commits in the current branch)")
  parser.add_option("-V", "--verbose", dest="verbose", default=False, action='store_true', help="be verbose about potential issues (warnings)")

  (options, args) = parser.parse_args()

  if options.version:
    if not VERSION_RE.match(options.version):
      parser.error("input version has to conform to the format M.m.p[ld] (where M, m, p, d are non-negative integers, l may be a, b or c")

  if options.count is not None and options.count < 0:
    parser.error("count has to be greater or equal zero")

  if args:
    parser.error("this program does not accept positional parameters")

  # change directories to my parent's
  os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0]))))

  branch = git_current_branch(options.verbose)

  if options.version is None:
    options.version = package_version(options.verbose)

    if options.version:
      # the directory in which this program is, is versioned
      print(options.version)
      sys.exit(0)

    if branch is not None:
      options.version =  git_next_version(branch, options.verbose)

  if options.count is None:
    options.count = git_count(branch, options.verbose)

  if not options.version:
    print('unknown')
  elif options.letter and not StrictVersion(options.version).prerelease:
    final = options.version + options.letter + str(options.count)
    StrictVersion(final) #double-checks all is good
    print(final)
  else:
    final = options.version
    StrictVersion(final) #double-checks all is good
    print(final)

  sys.exit(0)
