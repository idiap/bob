#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Mon 13 Feb 18:34:21 2012 

"""Self-contained script to generate python executables.
"""

TEMPLATE_LOCAL = """#!%(python)s
from %(module)s import %(method)s as main
main()
"""

TEMPLATE_NON_LOCAL = """#!%(python)s
import os
import sys
prefix = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
pyver = 'python%%d.%%d' %% sys.version_info[0:2]
library_path = os.path.realpath(os.path.join(prefix, 'lib'))
python_path = os.path.realpath(os.path.join(library_path, pyver))

def prepend_path(var, value):
  if os.environ.has_key(var): 
    os.environ[var] = '%%s:%%s' %% (value, os.environ[var])
  else: os.environ[var] = value

if not os.environ.has_key("__BOB_WRAPPER__"):
  os.environ['__BOB_WRAPPER__'] = "1"
  %(osx_dyld)s
  prepend_path('PYTHONPATH', python_path)
  sys.argv.insert(0, sys.executable)
  os.execve(sys.executable, sys.argv, os.environ)

# Call the announced callable
from %(module)s import %(method)s as main
main()
"""

import os
import sys
import platform
import stat
import argparse

def main():
  """Main method, parse arguments and generate the template"""

  parser = argparse.ArgumentParser(description=__doc__)
      #epilog=__epilog__, formatter_class=argparse.RawDescriptionHelpFormatter

  parser.add_argument("--non-local", dest="non_local", default=False, 
      action="store_true", help="If set, make the installation of python scripts such that it can execute from non-default directories. By default we assume that python scripts are installed in places where extensions can be located automatically")
  
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
      'module': args.module,
      'method': args.method,
      }

  if platform.system() == 'Darwin':
    dictionary['osx_dyld'] = "prepend_path('DYLD_LIBRARY_PATH', library_path)"
  else:
    dictionary['osx_dyld'] = ""
  
  if args.non_local:
    f.write(TEMPLATE_NON_LOCAL % dictionary)
  else:
    f.write(TEMPLATE_LOCAL % dictionary)
  f.close()
  del f

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
