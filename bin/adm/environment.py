#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 08 Jul 2010 09:18:28 CEST 

"""This program is able to start a new shell session or program with all 
required Torch variables already set. It will respect your shell choices 
and add to it."""

import sys, os
import optparse
import subprocess

epilog = """
Examples:

  If you are unsure of what to do, just print the help message:
  $ %(prog)s --help

  If you want to setup in debug mode for the current architecture:
  $ %(prog)s --debug
  
  If you want to setup for an arbitrary architecture:
  $ %(prog)s --arch=linux-i686-release
  
  If you want to execute a program under a setup using this installation
  $ %(prog)s --execute="<path-to-executable> <executable-options>"
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

  default_arch = current_arch(debug=False)

  #checks externals to see if they exist, if they do, put it as default.
  default_externals = []
  idiap_externals = \
      os.path.join("/idiap/group/torch5spro/nightlies/externals/last",
          default_arch)
  if os.path.exists(idiap_externals):
    default_externals.append(idiap_externals)

  default_executable = os.environ['SHELL']

  parser = MyParser(prog=prog, description=__doc__, 
      epilog=epilog % {'prog': prog})
  parser.add_option("-a", "--arch",
                    action="store",
                    dest="arch",
                    default=current_arch(debug=False),
                    help="Changes the default architecture for the setup.",
                   )
  parser.add_option("-d", "--debug", 
                    action="store_const",
                    const=current_arch(debug=True),
                    dest="arch", 
                    help="Outputs settings to run against the debug build",
                   )
  parser.remove_option("--help")
  parser.add_option("-h", "--help",
                    action="store_true",
                    dest="help",
                    default=False,
                    help="If this option is active, I'll print the help message and exit with status 3.",
                    )
  parser.add_option("-e", "--external",
                    action="append",
                    dest="externals",
                    default=default_externals,
                    help="Prepends external paths containing library installations to the list of searched paths. Please note that this list is taken backwards. (defaults to %default)."
                    )
  parser.add_option("-v", "--verbose",
                    action="store_true",
                    dest="verbose",
                    default=False,
                    help="Prints messages during execution."
                    )
  parser.add_option("-x", "--execute",
                    action="store",
                    dest="executable",
                    default=default_executable,
                    help="Sets the name of the program to execute under the new environment (defaults to %default)."
                    )
  parser.add_option("-c", "--csh",
                    action="store_true",
                    dest="csh",
                    default=False,
                    help=optparse.SUPPRESS_HELP,
                   )

  options, arguments = parser.parse_args()

  if options.help:
    parser.print_help()
    parser.exit(status=3)

  #sets up the version
  options.version = version(self_root(), options.verbose)

  #reverses the externals input list so the appended user preferences come
  #first
  options.externals.reverse()

  #separates executable and exec_options
  if options.executable.find(' ') != -1:
    x = options.executable.split(' ', 1)
    options.executable = x[0]
    options.full_executable = [k for k in ' '.join(x).split(' ') if k]
  else:
    options.full_executable = [options.executable]

  return (options, arguments)

def version(dir, verbose):
  """Finds out my own version"""
  version_file = os.path.join(dir, '.version')
  retval = 'unknown'
  if os.path.exists(version_file):
    f = open(version_file, 'rt')
    lines = f.readlines()
    if lines: retval = lines[0].strip()
    f.close()
  else:
    if verbose: print "Version file '%s' does not exist" % version_file
  return retval

def self_root():
  """Finds out where I am installed."""
  return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def python_version(environment):
  """Finds out the python version taking into consideration the external paths
  already setup."""
  try:
    return subprocess.Popen(["python", "-c", "import sys; print 'python%d.%d' % sys.version_info[0:2]"], stdout=subprocess.PIPE, env=environment).communicate()[0].strip()
  except OSError, e:
    return None

class PathJoiner(object):
  "Builds a os.path.join alias with a predefined prefix."

  def __init__(self, d): 
    self.directory = d

  def __call__(self, *args):
    return os.path.join(self.directory, *args)

class PathManipulator(object):
  "Builds a dict[key].insert(0, arg) aliases and the such."

  def __init__(self, d):
    self.dictionary = d

  def prepend_one(self, key, value):
    if value in self.dictionary[key]:
      del self.dictionary[key][self.dictionary[key].index(value)]
    self.dictionary[key].insert(0, value)

  def append_one(self, key, value):
    if value in self.dictionary[key]:
      del self.dictionary[key][self.dictionary[key].index(value)]
    self.dictionary[key].append(value)

  def before(self, key, *args):
    if not self.dictionary.has_key(key): self.dictionary[key] = []
    elif not isinstance(self.dictionary[key], list): 
      #break-up and remove empty
      self.dictionary[key] = [v for v in self.dictionary[key].split(':') if v]
    for value in args: self.prepend_one(key, value)

  def after(self, key, *args):
    if not self.dictionary.has_key(key): self.dictionary[key] = []
    for value in args: self.append_one(key, value)

  def consolidate(self):
    """Returns a copy of the environment dict where all entries are strings."""
    retval = dict(self.dictionary)
    for key, value in retval.iteritems():
      if isinstance(value, list): retval[key] = ':'.join(value)
    return retval

def setup_external_dir(environment, directory, arch, verbose):
  """Sets up the environment for libraries and utilities installed under
  "directory"."""

  if not os.path.exists(directory):
    if verbose: print "External '%s' ignored -- not accessible!"
    return

  J = PathJoiner(directory)
  P = PathManipulator(environment)

  P.before('PATH', J('bin'))
  P.before('MANPATH', J('man'), J('share', 'man'))
  P.before('PKG_CONFIG_PATH', J('lib', 'pkgconfig'))
  P.before('LD_LIBRARY_PATH', J('lib'))
  if arch.split('-')[0] == 'macosx':
    P.before('DYLD_LIBRARY_PATH', J('lib'))
  P.before('CMAKE_PREFIX_PATH', directory)

  #We don't setup PYTHONPATH because you are supposed to have that
  #done when I use the python interpreter from that environment!

  matlabdir = J('matlab')
  if os.path.exists(matlabdir):
    P.before('MATLABDIR', matlabdir)
    P.before('MATLAB_JAVA', '/usr/lib/jvm/java-6-sun/jre')
    P.before('AWT_TOOLKIT', 'MToolKit')
    MJ = PathJoiner(matlabdir)
    P.after('PATH', MJ('bin'))
    matlab_ld = MJ('bin', 'glnx86')
    if arch.split('-')[1] == 'x86_64': matlab_ld = MJ('bin', 'glnxa64')
    environment['MATLAB_LIBRARY_PATH'] = environment['LD_LIBRARY_PATH']
    P.before('MATLAB_LIBRARY_PATH', matlab_ld)

def shell_str(env, value, csh=False):
  """Outputs the correct environment set string."""
  if csh: return 'setenv %s "%s";' % (env, value)
  else: return 'export %s="%s";' % (env, value)

def shell_echo(value):
  """Outputs and echo message"""
  return 'echo "%s";' % value

def generate_environment(options):
  """Returns a list of environment variables that should make part a to-be 
  generated shell."""

  envdict = dict(os.environ) #copies

  #loads all external environment
  for k in options.externals:
    setup_external_dir(envdict, k, options.arch, options.verbose)

  J = PathJoiner(self_root()) #root_dir
  JIA = PathJoiner(J('install', options.arch)) #install_dir
  P = PathManipulator(envdict)

  pyver = python_version(P.consolidate())
  if pyver:
    if options.verbose: print "Python version detected: %s" % pyver
  else:
    if options.verbose: print "No python interpreter detected - using current"
    pyver = 'python%d.%d' % sys.version_info[0:2]

  P.before('PATH', J('bin'), JIA('bin'))
  P.before('PYTHONPATH', JIA('lib', pyver), JIA('lib'))
  P.before('LD_LIBRARY_PATH', JIA('lib'))
  if options.arch.split('-')[0] == 'macosx': # we are under OSX
    P.before('DYLD_LIBRARY_PATH', JIA('lib'))
  P.before('CMAKE_PREFIX_PATH', JIA('share', 'cmake'))
  P.before('TORCH_SCHEMA_PATH', JIA('share', 'torch', 'schema'))

  return P.consolidate()
