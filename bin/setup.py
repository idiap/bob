#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 08 Jul 2010 09:18:28 CEST 

"""A generic setup system for Torch"""

import sys, os
import optparse

epilog = """
Examples:

  If you are unsure of what to do, just print the help message:
  $ %(prog)s --help

  If you want to setup in debug mode for the current architecture:
  $ %(prog)s --debug
  
  If you want to setup for an arbitrary architecture:
  $ %(prog)s --arch=linux-i686-release
  
  If you want to show what would be done:
  $ %(prog)s --simulate
"""

def current_arch(debug):
  """Calculates the current arch"""
  import platform
  
  base = platform.system().lower()
  if base == 'darwin': 
    base = 'macosx' #change name to something nicer and easy to identify

  arch = platform.architecture()[0]
  if arch == '32bit': arch = 'i686'
  elif arch == '64bit': arch = 'x86_64'

  btype = 'release'
  if debug: btype = 'debug'
  return '%s-%s-%s' % (base, arch, btype) 

def parse_args():
  """Parses the command line input."""

  class MyParser(optparse.OptionParser):
    """Overwites the format_epilog() so we keep newlines..."""
    def format_epilog(self, formatter):
      return self.epilog

  prog =  os.path.basename(sys.argv[0])
  if len(sys.argv) > 1 and sys.argv[1][0] != '-': 
    prog = os.path.basename(sys.argv[1])

  parser = MyParser(prog=prog, description=__doc__, 
      epilog=epilog % {'prog': prog})
  parser.add_option("-a", "--arch",
                    action="store",
                    dest="arch",
                    default=current_arch(debug=False),
                    help="Changes the default architecture for the setup.",
                   )
  parser.add_option("-c", "--csh", 
                    action="store_true",
                    dest="csh", 
                    default=False,
                    #help="Generates output for csh compatible shells",
                    help=optparse.SUPPRESS_HELP,
                   )
  parser.add_option("-d", "--debug", 
                    action="store_const",
                    const=current_arch(debug=True),
                    dest="arch", 
                    help="Outputs settings to run against the debug build",
                   )
  parser.add_option("-n", "--check-options",
                    action="store_true",
                    dest="checker",
                    default=False,
                    #help="If this option is active, I'll check if everything is alright and exit with a status of 0 if so.",
                    help=optparse.SUPPRESS_HELP,
                    )
  parser.remove_option("--help")
  parser.add_option("-h", "--help",
                    action="store_true",
                    dest="help",
                    default=False,
                    help="If this option is active, I'll print the help message and exit with status 3.",
                    )
  parser.add_option("-s", "--simulate",
                    action="store_true",
                    dest="simulate",
                    default=False,
                    help="If this option is active, I'll show what the shell would do to setup and exit with status 4.",
                    )
  parser.add_option("-e", "--externals-setup",
                    action="store",
                    dest="externals_setup",
                    default="/idiap/group/torch5spro/nightlies/externals/tools/setup.py",
                    help="Set this to a different path if your externals are installed somewhere else (defaults to %default)."
                    )
  parser.add_option("-v", "--externals-version",
                    action="store",
                    dest="externals_version",
                    default="last",
                    help="Set this to a different value if you want a different externals version (defaults to %default)."
                   )

  options, arguments = parser.parse_args()

  if options.help: 
    parser.print_help()
    parser.exit(status=3)

  #sets up the version
  options.version = version(self_root())

  return (options, arguments)

def version(dir):
  """Finds out my own version"""
  version_file = os.path.join(dir, '.version')
  retval = 'unknown'
  if os.path.exists(version_file):
    f = open(version_file, 'rt')
    lines = f.readlines()
    if lines: retval = lines[0].strip()
    f.close()
  return retval

def self_root():
  """Finds out where I am installed."""
  return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

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

def shell_echo(value):
  """Outputs and echo message"""
  return 'echo "%s";' % value

def load_externals(options):
  """Loads environment from the externals of choice."""
  if not os.path.exists(options.externals_setup) or \
      not options.externals_setup: return {}
  class Opt(object): pass
  opt = Opt()
  opt.arch = externals_arch(options)
  opt.version = options.externals_version
  
  # this magic will load the "externals" setup module
  import imp; extsetup = imp.load_source('extsetup', options.externals_setup)

  print shell_echo("Setting up externals '%s' for architecture '%s'..." % \
      (options.externals_version, externals_arch(options)))

  # this bit will execute the setup for the externals
  return extsetup.setup_this(opt)

def generate_environment(options):
  """Returns a list of environment variables that need setting."""

  external_env = load_externals(options)

  base_dir = os.path.join(self_root(), 'install')
  install_dir = os.path.join(base_dir, options.arch)

  all = {}

  path = os.environ.get('PATH', '')
  path = path_remove(path, '.') #security concern
  path = path_remove_if_startswith(path, base_dir)
  path = path_add(path, os.path.join(self_root(), 'bin'))
  path = path_add(path, os.path.join(install_dir, 'bin'))
  all['PATH'] = path

  pythonpath = os.environ.get('PYTHONPATH', '')
  libdir = os.path.join(install_dir, 'lib')
  pythonpath = path_remove_if_startswith(pythonpath, base_dir) 
  python_version = 'python%d.%d' % (sys.version_info[0], sys.version_info[1])
  pythonpath = path_add(pythonpath, os.path.join(libdir, python_version))
  pythonpath = path_add(pythonpath, libdir)
  all['PYTHONPATH'] = pythonpath

  uname = os.uname()
  if options.arch.split('-')[0] == 'macosx': # we are under OSX
    dyld_library_path = os.environ.get('DYLD_LIBRARY_PATH', '')
    dyld_library_path = path_remove_if_startswith(dyld_library_path, base_dir)
    dyld_library_path = path_add(dyld_library_path, libdir)
    # this is for taking into consideration our python mac ports installation
    all['DYLD_LIBRARY_PATH'] = dyld_library_path

  ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
  ld_library_path = path_remove_if_startswith(ld_library_path, base_dir)
  ld_library_path = path_add(ld_library_path, libdir)
  all['LD_LIBRARY_PATH'] = ld_library_path

  # this is for cmake
  cmake_prefix_path = os.environ.get('CMAKE_PREFIX_PATH', '')
  cmakedir = os.path.join(install_dir, 'share', 'cmake')
  cmake_prefix_path = path_remove_if_startswith(cmake_prefix_path, base_dir)
  cmake_prefix_path = path_add(cmake_prefix_path, cmakedir)
  all['CMAKE_PREFIX_PATH'] = cmake_prefix_path

  # updates with keys found in the externals, but not touched by our procedure
  for k in external_env.keys():
    if not all.has_key(k): all[k] = external_env[k]

  return all

def setup_this(options):
  """Sets up the current python application"""
  env = generate_environment(options)
  for k, v in env.iteritems():
    if k == 'PYTHONPATH':
      for i in v.split(':'): sys.path.append(i)
    os.environ[k] = v
  return env

def externals_arch(options):
  return '-'.join(options.arch.split('-')[0:2]) + '-release'

if __name__ == '__main__':

  options, arguments = parse_args()

  # do not execute anything else, just exit
  if options.checker and not options.simulate: sys.exit(0)

  #echo what will be setup
  print shell_echo("Setting up torch5spro '%s' for architecture '%s'..." % \
      (options.version, options.arch))

  for k, v in generate_environment(options).iteritems():
    print shell_str(k, v, options.csh)

  if options.simulate: sys.exit(4)

  sys.exit(0)
