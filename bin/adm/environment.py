#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 08 Jul 2010 09:18:28 CEST 

"""This program is able to start a new shell session or program with all 
required Torch variables already set. It will respect your shell choices 
and add to it."""

import sys, os, time
import optparse
import subprocess

epilog = """
Examples:

  To create a new (sub) shell with torch configured just type:
  $ %(prog)s

  Leaving that shell will bring you to the place you are now, clearing all of
  your environment.

  If you are unsure of what to do, just print the help message:
  $ %(prog)s --help

  If you want to setup in debug mode for the current architecture:
  $ %(prog)s --debug
  
  If you want to setup for an arbitrary architecture:
  $ %(prog)s --arch=linux-i686-release
  
  If you want to execute a program under a setup using this installation.
  Please note that the two dashes without options halt option processing and
  enable you to give options to the executable you want to run:
  $ %(prog)s --debug -- <path-to-executable> <executable-options>

  No need for quotes or the such
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

def parse_args(argv):
  """Parses the command line input."""

  class MyParser(optparse.OptionParser):
    """Overwites the format_epilog() so we keep newlines..."""
    def format_epilog(self, formatter):
      return self.epilog

  prog =  os.path.basename(argv[0])
  if len(argv) > 1 and argv[1][0] != '-':
    prog = os.path.basename(argv[1])

  default_arch = current_arch(debug=False)

  #checks externals to see if they exist, if they do, put it as default.
  default_externals = []
  idiap_externals = \
      os.path.join("/idiap/group/torch5spro/nightlies/externals/last",
          default_arch)
  if os.path.exists(idiap_externals):
    default_externals.append(idiap_externals)

  parser = MyParser(prog=prog, description=__doc__, 
      epilog=epilog % {'prog': prog,})
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
                    action="count",
                    default=0,
                    dest="verbose",
                    help="Prints messages during execution. Using this option multiple times increases the verbosity level"
                    )
  parser.add_option("-c", "--csh",
                    action="store_true",
                    dest="csh",
                    default=False,
                    help=optparse.SUPPRESS_HELP,
                   )
  parser.add_option("-m", "--environment-manipulation",
                    action="store_true",
                    dest="env_manipulation",
                    default=False,
                    help="Makes this program change your executable behavior to be more torch friendly, if it can"
                   )
  parser.add_option("-r", "--root-dir",
                    action="store",
                    dest="root_dir",
                    default=self_root(),
                    help="Switch to a different base installation (defaults to %default)",
                   )
                    

  options, arguments = parser.parse_args(argv[1:])

  if options.help:
    parser.print_help()
    parser.exit(status=3)

  #sets up the version
  options.version = version(options.root_dir, options.verbose)

  #reverses the externals input list so the appended user preferences come
  #first
  options.externals.reverse()

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
    if verbose >= 3: print("Version file '%s' does not exist" % version_file)
  return retval

def self_root():
  """Finds out where I am installed."""
  return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def python_version(environment):
  """Finds out the python version taking into consideration the external paths
  already setup."""
  try:
    retval = subprocess.Popen(["python", "-c", "import sys; print 'python%d.%d' % sys.version_info[0:2]"], stdout=subprocess.PIPE, env=environment).communicate()[0].strip()
    return str(retval)
  except OSError as e:
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

  def set(self, key, value):
    self.dictionary[key] = value

  def prepend_one(self, key, value):
    if value in self.dictionary[key]:
      del self.dictionary[key][self.dictionary[key].index(value)]
    self.dictionary[key].insert(0, value)

  def append_one(self, key, value):
    if value in self.dictionary[key]:
      del self.dictionary[key][self.dictionary[key].index(value)]
    self.dictionary[key].append(value)

  def before(self, key, *args):
    if key not in self.dictionary: self.dictionary[key] = []
    elif not isinstance(self.dictionary[key], list): 
      #break-up and remove empty
      self.dictionary[key] = [v for v in self.dictionary[key].split(os.pathsep) if v]
    for value in args: self.prepend_one(key, value)

  def after(self, key, *args):
    if key not in self.dictionary: self.dictionary[key] = []
    for value in args: self.append_one(key, value)

  def consolidate(self):
    """Returns a copy of the environment dict where all entries are strings."""
    retval = dict(self.dictionary)
    for key, value in retval.items():
      if isinstance(value, list): retval[key] = os.pathsep.join(value)
    return retval

def setup_external_dir(environment, directory, arch, verbose):
  """Sets up the environment for libraries and utilities installed under
  "directory"."""

  if not os.path.exists(directory):
    if verbose >= 3: print("External '%s' ignored -- not accessible!")
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

  root = untemplatize_path(options.root_dir, options)

  J = PathJoiner(root) #root_dir
  JIA = PathJoiner(J('install', options.arch)) #install_dir
  P = PathManipulator(envdict)

  pyver = python_version(P.consolidate())
  if pyver:
    if options.verbose >= 3: print("Python version detected: %s" % pyver)
  else:
    if options.verbose >= 3: print("No python interpreter detected - using current")
    pyver = 'python%d.%d' % sys.version_info[0:2]

  P.before('PATH', J('bin'), JIA('bin'))
  P.before('PYTHONPATH', JIA('lib', pyver), JIA('lib'))
  P.before('LD_LIBRARY_PATH', JIA('lib'))
  if options.arch.split('-')[0] == 'macosx': # we are under OSX
    P.before('DYLD_LIBRARY_PATH', JIA('lib'))
  P.before('CMAKE_PREFIX_PATH', JIA('share', 'cmake'))
  P.before('TORCH_SCHEMA_PATH', JIA('share', 'torch', 'schema'))

  #this will place a few TORCH_ variables into the game, so the user can make
  #adjust its shell behavior accordinly, if he/she ever wants it.

  P.set('TORCH_INSTALL_DIR', root)
  P.set('TORCH_VERSION', version(root, False))
  P.set('TORCH_PLATFORM', options.arch)

  return P.consolidate()

def set_prompt(arguments, environ):
  """Adjusts the user prompt depending on the executable of choice. This will
  also depend on if we can make a better setting of the user environment or
  not."""
  executable = os.path.basename(arguments[0])
  J = PathJoiner(environ['TORCH_INSTALL_DIR'])
  JIA = PathJoiner(J('install', environ['TORCH_PLATFORM'])) #install_dir

  #this should be a gigantic switch covering all cases we know about:
  if executable in ('bash',):
    rcfile = J('bin', 'rcfiles', 'bash')
    arguments.extend(('--rcfile', rcfile))
    return

def untemplatize_path(path, options):
  """Removes string templates that may have been inserted into the path
  descriptor and returns a fully resolved string.
  """
  replacements = {
      'name': 'torch5spro',
      'version': 'alpha',
      'date': time.strftime("%d.%m.%Y"),
      'weekday': time.strftime("%A").lower(),
      'platform': options.arch,
      'root-dir': options.root_dir,
      }
  retval = path % replacements
  if retval.find('%(') != -1:
    raise RuntimeError("Cannot fully expand path `%s'" % retval)
  return retval
