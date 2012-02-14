#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Mon 13 Feb 18:34:21 2012 

"""Self-contained script to generate python executables.
"""

TEMPLATE = """#!%(python)s

import os
import sys
import platform

prefix = os.path.realpath(os.path.dirname(__file__))
pyver = 'python%%d.%%d' %% sys.version_info[0:2]
library_path = os.path.realpath(os.path.join(prefix, 'lib'))
python_path = os.path.realpath(os.path.join(library_path, pyver))

if not os.environ.has_key("BOB_VERSION"):

  # replace the interpreter in this case, with the proper environment
  os.environ['BOB_VERSION'] = '%(version)s'

  # dlopen() (used by python) on OSX does not look-up the rpath
  if platform.system() == 'Darwin':
    if os.environ.has_key('DYLD_LIBRARY_PATH'):
      os.environ['DYLD_LIBRARY_PATH'] = '%%s:%%s' %% (library_path, os.environ['DYLD_LIBRARY_PATH'])
    else:
      os.environ['DYLD_LIBRARY_PATH'] = library_path

  if os.environ.has_key('PYTHONPATH'):
    os.environ['PYTHONPATH'] = '%%s:%%s' %% (python_path, os.environ['PYTHONPATH'])
  else:
    os.environ['PYTHONPATH'] = python_path

  # replace interpreter with a new instance embedded in the new environment
  # Note: execle() will never return
  os.execle(sys.executable, sys.executable, os.path.realpath(__file__), os.environ)

# Call the announced callable
from %(module)s import %(method)s as main
main()
"""

import os
import sys
import stat
import argparse

def main():
  """Main method, parse arguments and generate the template"""

  parser = argparse.ArgumentParser(description=__doc__)
      #epilog=__epilog__, formatter_class=argparse.RawDescriptionHelpFormatter

  parser.add_argument("version", metavar="VERSION",
      help="The version of BOB to be set for this script")
  
  parser.add_argument("module", help="The python module name (with dots)",
      metavar="MODULE")
  
  parser.add_argument("method", metavar="METHOD",
      help="The python method name which implements the main function that will be called")

  parser.add_argument("output", metavar="FILE",
  help="The name of the output file that will be generated")
  
  args = parser.parse_args()

  destdir = os.path.dirname(args.output)
  if destdir:
    # Try creating the destination directory, does not fail if it exists
    try:
      if not os.path.exists(destdir): os.makedirs(destdir)
    except OSError as exc: # Python >2.5
      if exc.errno == errno.EEXIST: pass
      else: raise

  f = open(args.output, 'wt')
  
  dictionary = {
      'python': sys.executable,
      'version': args.version,
      'module': args.module,
      'method': args.method,
      }

  f.write(TEMPLATE % dictionary)
  f.close()
  del f

  # Print current environment
  for key, value in os.environ.iteritems():
    print "%s: %s" % (key, value)

  # Set execution bit, depending on the read mode for user, group and others
  mode = os.stat(args.output).st_mode
  if mode & stat.S_IRUSR: 
    os.chmod(args.output, mode | stat.S_IXUSR)
    mode = os.stat(args.output).st_mode
  if mode & stat.S_IRGRP: 
    os.chmod(args.output, mode | stat.S_IXGRP)
    mode = os.stat(args.output).st_mode
  if mode & stat.S_IROTH: 
    os.chmod(args.output, mode | stat.S_IXOTH)
    mode = os.stat(args.output).st_mode

if __name__ == '__main__':
  main()
