#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri 06 Aug 2010 08:39:26 CEST 

"""This script will build a full release of torch. It uses default parameters
for inplace builds, but can be easily customized to for out-of-place builds,
like the ones done for a nightly. The output can be captured and saved in log
files.
"""

import os, sys, subprocess, time, logging, tempfile, pprint

LOGGING_LEVELS = [
                  logging.DEBUG,
                  logging.INFO,
                  logging.WARNING,
                  logging.ERROR,
                  logging.CRITICAL,
                 ]
CURRENT_LOGGING_LEVEL = 2
logging.basicConfig(level=LOGGING_LEVELS[CURRENT_LOGGING_LEVEL], 
                    format="%(asctime)s | %(levelname)s | %(message)s")

def increase_verbosity(option, opt, value, parser):
  """Increases the current verbosity level for the logging module."""
  global CURRENT_LOGGING_LEVEL
  next = CURRENT_LOGGING_LEVEL - 1
  if next < 0: next = 0
  logger = logging.getLogger()
  logger.setLevel(LOGGING_LEVELS[next])
  CURRENT_LOGGING_LEVEL = next

def decrease_verbosity(option, opt, value, parser):
  """Decreases the current verbosity level for the logging module."""
  global CURRENT_LOGGING_LEVEL
  next = CURRENT_LOGGING_LEVEL + 1
  if next > (len(LOGGING_LEVELS)-1): next = len(LOGGING_LEVELS) - 1
  logger = logging.getLogger()
  logger.setLevel(LOGGING_LEVELS[next])
  CURRENT_LOGGING_LEVEL = next

def parse_args():
  """Parses the command line input."""
  import optparse
  
  #some defaults
  actions = ('all', 'build', 'documentation', 'test')
  build_types = ('release', 'debug') #default is #0
  pwd = os.path.realpath(os.curdir)
  default_prefix = os.path.join(pwd, 'install')
  default_doc_prefix = os.path.join('share', 'doc')
  default_build_prefix = os.path.join(pwd, 'build')
  default_log_prefix = os.path.join('logs')
  sources=os.path.realpath(os.path.dirname(os.path.dirname(sys.argv[0])))
  default_doxyfile = os.path.join(sources, 'Doxyfile')

  parser = optparse.OptionParser(description=__doc__)
  parser.add_option("-a", "--action", 
                    type="string",
                    action="store",
                    dest="action", 
                    default=actions[0],
                    metavar="(%s)" % '|'.join(actions),
                    help="what should I build (defaults to %default)",
                   )
  parser.add_option("--sources",
                    type="string",
                    action="store",
                    dest="source_dir",
                    default=sources,
                    metavar="DIR",
                    help="builds from this source tree (defaults to %default)",
                   )
  parser.add_option("--build-type", 
                    type="string",
                    action="store",
                    dest="build_type", 
                    default=build_types[0],
                    metavar="(%s)" % '|'.join(build_types),
                    help="defines the build type (defaults to %default)",
                   )
  parser.add_option("--build-prefix",
                    type="string",
                    action="store",
                    dest="build_prefix",
                    default=default_build_prefix,
                    metavar="DIR",
                    help="base build directory (defaults to %default)",
                   )
  parser.add_option("--install-prefix",
                    type="string",
                    action="store",
                    dest="prefix",
                    default=default_prefix,
                    metavar="DIR",
                    help="base install directory (defaults to %default)",
                   )
  parser.add_option("--documentation-prefix",
                    type="string",
                    action="store",
                    dest="doc_prefix",
                    default=default_doc_prefix,
                    metavar="DIR",
                    help="base install directory for documentation (defaults to %default). note: if a relative directory is given, it is taken w.r.t. the install prefix.",
                   )
  parser.add_option("--doxyfile",
                    action="store",
                    dest="doxyfile",
                    default=default_doxyfile,
                    metavar="FILE",
                    help="path of a doxygen file to be used as configuration basis (defaults to %default)",
                  )
  parser.add_option("-l", "--log-output",
                    action="store_true",
                    dest="save_output",
                    default=False,
                    help="store output into files",
                   )
  parser.add_option("--log-directory",
                    type="string",
                    action="store",
                    dest="log_prefix",
                    default=default_log_prefix,
                    metavar="DIR",
                    help="where to place the log files (defaults to %default).  note: if a relative directory is given, it is taken w.r.t. the build prefix.",
                   )
  parser.add_option("-j", "--jobs",
                    type="int",
                    action="store",
                    dest="jobs",
                    default=1,
                    metavar="INT(>=1)",
                    help="where possible, try to launch several jobs at once (defaults to %default)",
                   )
  parser.add_option("--static-linkage",
                    action="store_true",
                    dest="static_linkage",
                    default=False,
                    help="links executables with static libraries when possible",
                   )
  parser.add_option("-d", "--debug-build",
                    action="store_true",
                    dest="debug_build",
                    default=False,
                    help="executes all stages with lots of verbose output",
                   )
  parser.add_option("-q", "--quiet",
                    action="callback",
                    callback=decrease_verbosity,
                    help="decreases the current verbosity level",
                   )
  parser.add_option("-v", "--verbose",
                    action="callback",
                    callback=increase_verbosity,
                    help="increases the current verbosity level",
                   )
  
  options, args = parser.parse_args()

  #some error checking
  if options.action not in actions:
    parser.error("option --action has to be one of %s" % ", ".join(actions))
  if options.build_type not in build_types:
    parser.error("option --build-type has to be one of %s" % ", ".join(build_types))
  if options.jobs < 1:
    parser.error("option --jobs has to be equal or greater than 1")
  if options.debug_build and options.jobs > 1:
    logging.info("option --jobs will be reset to 1")
    options.jobs = 1
  if args:
    parser.error("this program does not accept positional arguments: %s" % args)

  #some fixing
  options.prefix = options.prefix.strip()
  if options.prefix and options.prefix[0] != os.path.sep:
    options.prefix = os.path.join(pwd, options.prefix) 
  options.build_prefix = options.build_prefix.strip()
  if options.build_prefix and options.build_prefix[0] != os.path.sep:
    options.build_prefix = os.path.join(pwd, options.build_prefix) 
  options.doc_prefix = options.doc_prefix.strip()
  options.log_prefix = options.log_prefix.strip()

  return options, args

def run(cmd, log, dir, prefix):
  logging.debug('Executing: %s' % ' '.join(cmd))
  if log:
    if not os.path.exists(dir): os.makedirs(dir)
    fname = os.path.join(dir, prefix) + '.txt'
    logging.debug('Output: %s' % fname)
    stdout = file(fname, 'wt')
    stderr = stdout 
  else: 
    stdout = sys.stdout
    stderr = sys.stderr
    logging.debug('Output: current terminal')
  start = time.time()
  p = subprocess.Popen(cmd, stdin=None, stdout=stdout, stderr=stderr)
  p.wait()
  total = time.time() - start
  if total < 1:
    total = '%d milliseconds' % (1000*total)
  elif total < 60:
    if total >= 2: total = '%d seconds' % total
    else: total = '%d second' % total
  else:
    total = total/60
    if total >= 2: total = '%d minutes' % total
    else: total = '%d minute' % total
  logging.debug('Time used: %s' % total)
  if log: stdout.close()
  return p.returncode

def cmake(option, build_dir, install_dir):
  """Builds the project using cmake"""
  logging.debug('Running cmake...')

  if not os.path.exists(build_dir): os.makedirs(build_dir)

  os.chdir(build_dir)

  cmake_options = {}
  cmake_options['-DCMAKE_BUILD_TYPE'] = option.build_type
  cmake_options['-DINSTALL_DIR'] = install_dir
  cmake_options['-DINCLUDE_DIR'] = os.path.join(install_dir, 'include')
  cmake_options['-DTORCH_LINKAGE'] = 'dynamic'
  if option.static_linkage: cmake_options['-DTORCH_LINKAGE'] = 'static'
  cmdline = ['cmake']
  if option.debug_build: cmdline.append('--debug-output')
  for k,v in cmake_options.iteritems(): cmdline.append('%s=%s' % (k, v))
  cmdline.append(option.source_dir)
  status = run(cmdline, option.save_output, option.log_prefix, cmdline[0])
  if status != 0:
    raise RuntimeError, '** ERROR: "cmake" did not work as expected.'
  logging.debug('Finished running cmake.')

def make(option, build_dir, target="all"):
  logging.debug('Running make %s...' % target)

  os.chdir(build_dir)

  cmdline = ['make', '--keep-going']
  if option.debug_build:
    cmdline.append('VERBOSE=1')
  else:
    cmdline.append('-j%d' % option.jobs)
  cmdline.append(target)
  status = run(cmdline, option.save_output, option.log_prefix, cmdline[0]+'_'+target)
  if status != 0:
    raise RuntimeError, '** ERROR: "make %s" did not work as expected.' % target
  logging.debug('Finished running make %s.' % target)

def doxygen(option):
  """Builds the project documentation using doxygen"""
  logging.debug('Running doxygen...')

  if not os.path.exists(option.doc_prefix): os.makedirs(option.doc_prefix)

  overwrite_options = {}
  overwrite_options['INPUT'] = os.path.join(option.source_dir, 'src')
  overwrite_options['STRIP_FROM_PATH'] = option.source_dir
  overwrite_options['OUTPUT_DIRECTORY'] = option.doc_prefix
  if option.debug_build: overwrite_options['QUIET'] = 'NO'
  extras = []
  for k,v in overwrite_options.iteritems(): extras.append('%s = %s\n' % (k, v))

  original = file(option.doxyfile, 'rt')
  lines = original.readlines() + extras
  original.close()
  (tmpfd, tmpname) = tempfile.mkstemp()
  tmpfile = os.fdopen(tmpfd, 'wt')
  tmpfile.writelines(lines)
  tmpfile.seek(0)
 
  cmdline = ['doxygen', tmpname]
  status = run(cmdline, option.save_output, option.log_prefix, cmdline[0])
  if status != 0:
    raise RuntimeError, '** ERROR: "doxygen" did not work as expected.'
  tmpfile.close()
  os.unlink(tmpname)

  #create a link from index.html to main.html 
  os.chdir(os.path.join(option.doc_prefix, 'html'))
  if not os.path.exists('main.html'):
    logging.debug("Generating symlink main.html -> index.html")
    os.symlink('index.html', 'main.html')

  logging.debug('Finished running doxygen.')
  
def status_log(option, build_dir, install_dir, platform, timing, problems):
  """Writes a pythonic status file in the root of the log directory."""
  cfname = os.path.join(option.log_prefix, 'status.py')
  cfile = file(cfname, 'wt')
  logging.debug('Writing status file at %s' % cfile)
  prog = os.path.basename(sys.argv[0])
  pp = pprint.PrettyPrinter(indent=2)
  lines = [
           "# Generated automatically by %s" % prog, 
           "# You can load this directly into python by import or execfile()",
           "",
          ]
  lines.append('uname = %s' % pp.pformat(os.uname()))
  lines.append('')
  lines.append('# input options as reader by the parser')
  exec('optdict = %s' % option) #some python magic
  lines.append('options = %s' % optdict)
  lines.append('')
  lines.append('# this is the place we used to build')
  lines.append('build_dir = \'%s\'' % build_dir)
  lines.append('')
  lines.append('# this is the place we installed after the build')
  lines.append('install_dir = \'%s\'' % install_dir)
  lines.append('')
  lines.append('# calculated automatically by %s' % prog)
  lines.append('platform = \'%s\'' % platform)
  lines.append('')
  lines.append('# start/end = time.time(), other entries are intervals in seconds.')
  lines.append('timing = %s' % pp.pformat(timing))
  lines.append('')
  lines.append('# this is the status of the run.')
  lines.append('status = %s' % pp.pformat(problems))
  cfile.write('\n'.join(lines))
  cfile.write('\n')
  cfile.close()
  logging.debug('Finished writing status file.')
  return cfname

if __name__ == '__main__':
  time_track = {'start': time.time()}
  problem_track = {}

  (option, args) = parse_args()

  #calculates platform
  uname = os.uname()
  platform = '%s-%s-%s' % (uname[0].lower(), uname[4].lower(), 
      option.build_type)

  build_dir = os.path.join(option.build_prefix, platform)
  install_dir = os.path.join(option.prefix, platform)
  if option.log_prefix and option.log_prefix[0] != os.path.sep:
    option.log_prefix = os.path.join(build_dir, option.log_prefix)
  if option.doc_prefix and option.doc_prefix[0] != os.path.sep:
    option.doc_prefix = os.path.join(install_dir, option.doc_prefix)

  logging.info('== build.py setup ==')
  logging.info("Action: %s" % option.action.upper())
  logging.info("Job count: %d" % option.jobs)
  if option.save_output:
    logging.info("Log output: YES, on %s" % option.log_prefix)
  else: logging.info("Log output: NO")
  if option.action in ('all', 'build', 'test'):
    logging.info("Platform: %s" % platform)
    if option.static_linkage: 
      logging.info("Executable linkage: STATIC")
    else: 
      logging.info("Executable linkage: SHARED")
    logging.info("Sources: %s" % option.source_dir)
    logging.info("Build directory: %s" % build_dir)
    logging.info("Install directory: %s" % install_dir)
  if option.action in ('all', 'documentation'):
    logging.info("Documentation directory: %s" % option.doc_prefix)
    logging.info("Doxyfile base configuration: %s" % option.doxyfile)
  if option.action in ('all', 'test'):
    logging.info("Run tests after build: YES")
  
  problem_track = {'configuration': ('success',)}

  #build
  proceed = True

  if option.action in ('all', 'build', 'test'):
    start = time.time()
    try:
      cmake(option, build_dir, install_dir)
      problem_track['cmake'] = ('success', )
    except Exception, e:
      problem_track['cmake'] = ('failed', '%s' % e)
      proceed = False
    time_track['cmake'] = time.time() - start 

    start = time.time()
    try:
      if proceed: 
        make(option, build_dir, 'all')
        problem_track['make_all'] = ('success', )
      else:
        problem_track['make_all'] = ('blocked',)
    except Exception, e:
      problem_track['make_all'] = ('failed', '%s' % e)
      proceed = False
    time_track['make_all'] = time.time() - start 

    start = time.time()
    try:
      if proceed:
        make(option, build_dir, 'install')
        problem_track['make_install'] = ('success', )
      else:
        problem_track['make_install'] = ('blocked',)
    except Exception, e:
      problem_track['make_all'] = ('failed', '%s' % e)
      proceed = False
    time_track['make_install'] = time.time() - start 
  
  #documentation
  if option.action in ('all', 'documentation'):
    start = time.time()
    try:
      doxygen(option)
      problem_track['doxygen'] = ('success', )
    except Exception, e:
      problem_track['doxygen'] = ('failed', '%s' % e)
    time_track['doxygen'] = time.time() - start
  
  #test
  if option.action in ('all', 'test'):
    start = time.time()
    try:
      if proceed:
        make(option, build_dir, 'test')
        problem_track['make_test'] = ('success',)
      else:
        problem_track['make_test'] = ('blocked',)
    except Exception, e:
      problem_track['make_test'] = ('failed', '%s' % e)
      proceed = False
    time_track['make_test'] = time.time() - start 

  time_track['end'] = time.time()

  if option.save_output: #save base status file
    cfname = status_log(option, build_dir, install_dir, platform, 
        time_track, problem_track)
    print 'Status saved at %s' % cfname

  if proceed: sys.exit(0)
  else: sys.exit(1)
