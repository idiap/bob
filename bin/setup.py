#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 08 Jul 2010 09:18:28 CEST 

"""A generic setup system for Torch"""

import sys, os

epilog = """
Examples:

  If you are unsure of what to do, just print the help message:
  $ %(prog)s --help

  If you want to setup your shell and you have either csh or tcsh:
  $ eval `%(prog)s --csh`

  If you want to setup your shell and you have one of the sh variants:
  $ eval `%(prog)s --sh`

  If you want to setup in debug mode:
  $ eval `%(prog)s --debug --sh`

  If you don't know which shell type you have:
  $ echo $SHELL
""" % {'prog': os.path.basename(sys.argv[0])}

def parse_args():
  """Parses the command line input."""
  import optparse

  class MyParser(optparse.OptionParser):
    """Overwites the format_epilog() so we keep newlines..."""
    def format_epilog(self, formatter):
      return self.epilog

  dir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
  
  parser = MyParser(description=__doc__, epilog=epilog)
  parser.add_option("-b", "--base-dir", 
                    action="store",
                    dest="dir", 
                    default=dir,
                    help="Sets the base directory to a different value (defaults to %default)",
                   )
  parser.add_option("-c", "--csh", 
                    action="store_true",
                    dest="csh", 
                    default=False,
                    help="Outputs settings for csh shells (csh|tcsh)",
                   )
  parser.add_option("-d", "--debug", 
                    action="store_true",
                    dest="debug", 
                    default=False,
                    help="Outputs settings to run against the debug build",
                   )
  options, arguments = parser.parse_args()

  options.dir = os.path.realpath(options.dir)

  return (options, arguments)

def path_remove(env, what):
  """Removes the 'what' components from the path environment 'env', if it
  exists."""
  p = env.split(':')
  if what in p: del p[p.index(what)]
  return ':'.join([k for k in p if k.strip()])

def path_remove_if_startswith(env, what):
  """Removes in path that starts with 'what' from the path environment 'env',
  if any exist."""
  p = env.split(':')
  to_delete = []
  for k in p:
    if k.find(what) == 0: to_delete.append(k)
  for k in to_delete: del p[p.index(k)]
  return ':'.join([k for k in p if k.strip()])

def path_add(env, what, preffix=True):
  """Affixes 'what' into path, verifying if there are no other copies of it
  around."""

  p = path_remove(env, what).split(':')
  if preffix: p.insert(0, what)
  else: p.append(what)
  return ':'.join([k for k in p if k.strip()])

def shell_str(env, value, csh=False):
  """Outputs the correct environment set string."""
  if csh: return 'setenv %s "%s";' % (env, value)
  else: return 'export %s="%s";' % (env, value)

def setup_python(all):
  """Sets up a python application"""
  for k, v in all: os.environ[k] = v

def main(dir, debug):
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
  path = path_add(path, os.path.join(dir, 'bin'))
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

  # this is for cmake
  cmake_prefix_path = os.environ.get('CMAKE_PREFIX_PATH', '')
  cmakedir = os.path.join(install_dir, 'share', 'cmake')
  cmake_prefix_path = path_remove_if_startswith(cmake_prefix_path, base_dir)
  cmake_prefix_path = path_add(cmake_prefix_path, cmakedir)
  all.append(('CMAKE_PREFIX_PATH', cmake_prefix_path))

  return all

if __name__ == '__main__':

  options, arguments = parse_args()

  all = main(options.dir, options.debug)
  
  for k, v in all: print shell_str(k, v, options.csh)
