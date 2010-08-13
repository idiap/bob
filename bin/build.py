#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri 06 Aug 2010 08:39:26 CEST 

"""This script will build a full release of torch. It uses default parameters
for inplace builds, but can be easily customized to for out-of-place builds,
like the ones done for a nightly. The output can be captured and saved in log
files.
"""

import os
import sys 
import time
import logging

# Imports our admin toolkit
sys.path.append(os.path.realpath(os.path.dirname(sys.argv[0])))
import adm

def parse_args():
  """Parses the command line input."""
  import optparse
  
  #some defaults
  actions = ('all', 'build', 'documentation', 'test', 'depfigure')
  build_types = ('release', 'debug') #default is #0
  pwd = os.path.realpath(os.curdir)
  default_install_prefix = os.path.join(pwd, 'install')
  default_doc_prefix = os.path.join('share', 'doc')
  default_build_prefix = os.path.join(pwd, 'build')
  default_log_prefix = os.path.join('logs')
  sources=os.path.realpath(os.path.dirname(os.path.dirname(sys.argv[0])))
  sources=os.path.join(sources, 'src')
  default_doxyfile = os.path.join(sources, 'doc', 'Doxyfile')

  #our gigantic list of options...
  parser = optparse.OptionParser(description=__doc__)
  parser.add_option("-a", "--action", type="string", action="store", 
      dest="action", default=actions[0], metavar="(%s)" % '|'.join(actions),
      help="what should I build (defaults to %default)",
      )
  parser.add_option("--sources", type="string", action="store",
      dest="source_dir", default=sources, metavar="DIR",
      help="builds from this source tree (defaults to %default)",
      )
  parser.add_option("--build-type", type="string", action="store",
      dest="build_type", default=build_types[0],
      metavar="(%s)" % '|'.join(build_types),
      help="defines the build type (defaults to %default)",
      )
  parser.add_option("--build-prefix", type="string", action="store",
      dest="build_prefix", default=default_build_prefix, metavar="DIR",
      help="base build directory (defaults to %default)",
      )
  parser.add_option("--install-prefix", type="string", action="store",
      dest="install_prefix", default=default_install_prefix,
      metavar="DIR", help="base install directory (defaults to %default)",
      )
  parser.add_option("--documentation-prefix", type="string", action="store",
      dest="doc_prefix", default=default_doc_prefix, metavar="DIR",
      help="base install directory for documentation (defaults to %default). note: if a relative directory is given, it is taken w.r.t. the install prefix.",
      )
  parser.add_option("--doxyfile", action="store", dest="doxyfile",
      default=default_doxyfile, metavar="FILE",
      help="path of a doxygen file to be used as configuration basis (defaults to %default)",
      )
  parser.add_option("-l", "--log-output", action="store_true",
      dest="save_output", default=False, help="store output into files",
      )
  parser.add_option("--log-directory", type="string", action="store",
      dest="log_prefix", default=default_log_prefix, metavar="DIR",
      help="where to place the log files (defaults to %default).  note: if a relative directory is given, it is taken w.r.t. the build prefix.",
      )
  parser.add_option("-j", "--jobs", type="int", action="store", dest="jobs",
      default=1, metavar="INT(>=1)",
      help="where possible, try to launch several jobs at once (defaults to %default)",
      )
  parser.add_option("--static-linkage", action="store_true",
      dest="static_linkage", default=False,
      help="links executables with static libraries when possible",
      )
  parser.add_option("-d", "--debug-build", action="store_true",
      dest="debug_build", default=False,
      help="executes all stages with lots of verbose output",
      )
  parser.add_option("-q", "--quiet", action="callback",
      callback=adm.build.decrease_verbosity,
      help="decreases the current verbosity level",
      )
  parser.add_option("-v", "--verbose", action="callback",
      callback=adm.build.increase_verbosity,
      help="increases the current verbosity level",
      )
  parser.add_option("-V", "--version", action="store", dest="version",
      default="Version ?.?", metavar="VERSION",
      help="if it makes sense, choose a version name that will be used to mark the project documentation, otherwise, leave it unassigned"
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
  options.install_prefix = options.install_prefix.strip()
  if options.install_prefix and options.install_prefix[0] != os.path.sep:
    options.install_prefix = os.path.join(pwd, options.install_prefix) 
  options.build_prefix = options.build_prefix.strip()
  if options.build_prefix and options.build_prefix[0] != os.path.sep:
    options.build_prefix = os.path.join(pwd, options.build_prefix) 
  options.doc_prefix = options.doc_prefix.strip()
  options.log_prefix = options.log_prefix.strip()

  options.platform = adm.build.platform(options)
 
  #we suffix the platform on the build and installation directories
  options.build_prefix = os.path.join(options.build_prefix, options.platform)
  options.install_prefix = os.path.join(options.install_prefix, options.platform)

  #the log prefix we take w.r.t. the build prefix, if the user has not
  #insisted...
  if options.log_prefix and options.log_prefix[0] != os.path.sep:
    options.log_prefix = os.path.join(options.build_prefix, options.log_prefix)
  if options.doc_prefix and options.doc_prefix[0] != os.path.sep:
    options.doc_prefix = os.path.join(options.install_prefix, options.doc_prefix)

  #printing, if the user wants some debug information.
  logging.info('== build.py setup ==')
  logging.info("Action: %s" % options.action.upper())
  logging.info("Job count: %d" % options.jobs)
  if options.save_output:
    logging.info("Log output: YES, on %s" % options.log_prefix)
  else: logging.info("Log output: NO")
  if options.action in ('all', 'build', 'test'):
    logging.info("Platform: %s" % options.platform)
    if options.static_linkage: 
      logging.info("Executable linkage: STATIC")
    else: 
      logging.info("Executable linkage: SHARED")
    logging.info("Sources: %s" % options.source_dir)
    logging.info("Build directory: %s" % options.build_prefix)
    logging.info("Install directory: %s" % options.install_prefix)
  if options.action in ('all', 'documentation'):
    logging.info("Documentation directory: %s" % options.doc_prefix)
    logging.info("Doxyfile base configuration: %s" % options.doxyfile)
  if options.action in ('all', 'test'):
    logging.info("Run tests after build: YES")
  
  return options, args

if __name__ == '__main__':
  time_track = {'start': time.time()}
  problem_track = {}

  (options, args) = parse_args()

  problem_track = {'configuration': ('success',)}

  if options.action in ('all', 'build', 'test', 'depfigure'):
    phase = 'cmake'
    time_track[phase], problem_track[phase] = \
        adm.build.action(adm.build.cmake, options)

  if options.action in ('all', 'build', 'test'):
    phase = 'compile'
    if problem_track['cmake'][0] == 'success':
      time_track[phase], problem_track[phase] = \
          adm.build.action(adm.build.make, options, 'all')
    else:
      time_track[phase] = 0
      problem_track[phase] = ('blocked',)

  if options.action in ('all', 'build', 'test'):
    phase = 'install'
    if problem_track['compile'][0] == 'success':
      time_track[phase], problem_track[phase] = \
          adm.build.action(adm.build.make, options, 'install')
    else:
      time_track[phase] = 0
      problem_track[phase] = ('blocked',)

  #dependency figure, depends only on cmake passing...
  if options.action in ('all', 'depfigure'):
    phase = 'depfigure'
    if problem_track['cmake'][0] == 'success':
      time_track[phase], problem_track[phase] = \
          adm.build.action(adm.build.dot, options)
    else:
      time_track[phase] = 0
      problem_track[phase] = ('blocked',)

  #documentation, depends on nothing else
  if options.action in ('all', 'documentation'):
    phase = 'documentation'
    time_track[phase], problem_track[phase] = \
        adm.build.action(adm.build.doxygen, options)

  #test, depends on install
  if options.action in ('all', 'test'):
    phase = 'test'
    if problem_track['install'][0] == 'success':
      time_track[phase], problem_track[phase] = \
          adm.build.action(adm.build.make, options, 'test')
    else:
      time_track[phase] = 0
      problem_track[phase] = ('blocked',)

  time_track['end'] = time.time()

  if options.save_output: #save base status file
    cfname = adm.build.status_log(options, time_track, problem_track)
    print 'Status saved at %s' % cfname

  success = max([k[0] == 'success' for k in problem_track])
  if not success: sys.exit(1)

  sys.exit(0)
