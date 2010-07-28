#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 08 Jul 2010 09:18:28 CEST 

"""A generic setup system for Torch5

usage: setup.py [options]
  options:
  -h|-?|--help Prints this help message
  -d|--debug   Outputs settings to run against the debug build
  -c|--csh     Outputs settings for csh shells (csh|tcsh)
  -s|--sh      Outputs settings for sh shells (sh|bash|zsh|... etc)

examples:

  If you are unsure of what to do, just print the help message:
  $ ./setup.py --help

  If you want to setup your shell and you have either csh or tcsh:
  $ eval `./setup.py --csh`

  If you want to setup your shell and you have one of the sh variants:
  $ eval `./setup.py --sh`

  If you want to setup in debug mode:
  $ eval `./setup.py --debug --sh`

  If you don't know which shell type you have:
  $ echo $SHELL
"""

import os, sys

def path_remove(env, what):
  """Removes the 'what' components from the path environment 'env', if it
  exists."""
  p = env.split(':')
  if what in p: del p[p.index(what)]
  return ':'.join(p)

def path_remove_if_startswith(env, what):
  """Removes in path that starts with 'what' from the path environment 'env',
  if any exist."""
  p = env.split(':')
  to_delete = []
  for k in p:
    if k.find(what) == 0: to_delete.append(k)
  for k in to_delete: del p[p.index(k)]
  return ':'.join(p)

def path_add(env, what, preffix=True):
  """Affixes 'what' into path, verifying if there are no other copies of it
  around."""

  p = path_remove(env, what).split(':')
  if preffix: p.insert(0, what)
  else: p.append(what)
  return ':'.join(p)

def shell_str(env, value, csh=False):
  """Outputs the correct environment set string."""
  if csh: return 'setenv %s "%s";' % (env, value)
  else: return 'export %s="%s";' % (env, value)

def main(dir, debug, csh):
  """Searches for the parent shell type and outputs the correct environment
  settings for that."""

  uname = os.uname()
  platform = '%s-%s' % (uname[0].lower(), uname[4].lower())
  if debug: platform += '-debug'
  else: platform += '-release'

  base_dir = os.path.join(dir, 'install')
  install_dir = os.path.join(base_dir, platform)

  all = []

  path = os.environ.get('PATH', '')
  path = path_remove(path, '.') #security concern
  path = path_remove_if_startswith(path, base_dir)
  path = path_add(path, os.path.join(base_dir, 'bin'))
  path = path_add(path, os.path.join(install_dir, 'bin'))
  all.append(('PATH', path))

  pythonpath = os.environ.get('PYTHONPATH', '')
  libdir = os.path.join(install_dir, 'lib')
  pythonpath = path_remove_if_startswith(pythonpath, base_dir) 
  python_version = 'python%d.%d' % (sys.version_info[0], sys.version_info[1])
  pythonpath = path_add(pythonpath, os.path.join(libdir, python_version))
  pythonpath = path_add(pythonpath, libdir)
  all.append(('PYTHONPATH', pythonpath))

  if uname[0].lower() == 'darwin': # we are under OSX
    dyld_library_path = os.environ.get('DYLD_LIBRARY_PATH', '')
    dyld_library_path = path_remove_if_startswith(dyld_library_path, base_dir)
    dyld_library_path = path_add(dyld_library_path, libdir)
    # this is for taking into consideration our python mac ports installation
    all.append(('DYLD_LIBRARY_PATH', dyld_library_path))

  ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
  ld_library_path = path_remove_if_startswith(ld_library_path, base_dir)
  ld_library_path = path_add(ld_library_path, libdir)
  all.append(('LD_LIBRARY_PATH', ld_library_path))

  for k, v in all: print shell_str(k, v, csh)

if __name__ == '__main__':
  dir = os.path.realpath(os.path.dirname(sys.argv[0]))
  debug = False
  csh = False

  if len(sys.argv) == 1:
    print __doc__
    sys.exit(1)
  if len(sys.argv) > 1:
    if sys.argv[1] in ('-h', '-?', '--help'):
      print __doc__
      sys.exit(1)
    for arg in sys.argv[1:]:
      if arg in ('-d', '--debug'): debug = True
      elif arg in ('-c', '--csh'): csh = True
      elif arg in ('-s', '--sh'): csh = False
      else:
        print __doc__
        sys.exit(1)

  main(dir, debug, csh)
