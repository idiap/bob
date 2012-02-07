#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# André Anjos <andre.anjos@idiap.ch>
# Thu Jan 20 18:08:37 2011 +0100
#
# Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""This script is used by Buildbot to build full bob releases. It knows how
to handle all construction phases. If you are looking for a way to build bob
locally, look at build.py or aliases "release.sh" and "debug.sh".

This program should be prefixed in a bob-enabled shell environment. You can
create such an environment by calling this program like this:

  <bob-root-installation>/bin/shell.py -x "build.py <command>"
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
  actions = ('cmake', 'dot', 'make_all', 'all', 'build', 'documentation',
      'docs', 'sphinx', 'doxygen', 'make_install', 'install', 'ctest', 'test',
      'make_clean', 'clean', 'mrproper', 'group_write', 'group_unwrite')
  build_types = ('release', 'debug') #default is #0
  build_blocks = ('all', 'cxx', 'python') #default is #0
  pwd = os.path.realpath(os.curdir)
  default_install_prefix = os.path.join(pwd, 'install', '%(platform)s')
  default_doc_prefix = os.path.join('%(install-prefix)s', 'share', 'doc')
  default_build_prefix = os.path.join(pwd, 'build', '%(platform)s')
  sources = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
  sources = os.path.join(sources, 'src')
  default_doxyfile = os.path.join(os.path.dirname(sources), 'doc', 'doxygen', 'Doxyfile')
  default_sphinxdir = os.path.join(os.path.dirname(sources), 'doc', 'sphinx')

  #our gigantic list of options...
  parser = optparse.OptionParser(description=__doc__)
  parser.add_option("-a", "--action", type="string", action="store", 
      dest="action", default=actions[0], metavar="(%s)" % '|'.join(actions),
      help="what should I build (defaults to %default)",
      )
  parser.add_option("--build-type", type="string", action="store",
      dest="build_type", default=build_types[0],
      metavar="(%s)" % '|'.join(build_types),
      help="defines the build type (defaults to %default)",
      )
  parser.add_option("-B", "--build-block", type="string", action="store",
      dest="build_block", default=build_blocks[0],
      metavar="(%s)" % '|'.join(build_blocks),
      help="defines which libraries to build (defaults to %default)",
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
  parser.add_option("--sphinxdir", action="store", dest="sphinxdir",
      default=default_sphinxdir, metavar="DIR",
      help="path of the sphinx directory to be used (defaults to %default)",
      )
  parser.add_option("-j", "--jobs", type="int", action="store", dest="jobs",
      default=1, metavar="INT(>=1)",
      help="where possible, try to launch several jobs at once (defaults to %default)",
      )
  parser.add_option("-G", "--graphviz", action="store_true", dest="graphviz",
      default=False,
      help="also generates the dependence graph in dot format while in cmake",
      )
  parser.add_option("--static-linkage", action="store_true",
      dest="static_linkage", default=False,
      help="links executables with static libraries when possible",
      )
  parser.add_option("-d", "--debug-build", action="store_true",
      dest="debug_build", default=False,
      help="executes all stages with lots of verbose output",
      )
  parser.add_option("-k", "--create-databases", action="store_true",
      dest="createdb", default=False,
      help="issues the create database commands during the test phase",
      )
  parser.add_option("-K", "--database-prefix", action="store",
      dest="db_prefix", default=None, metavar="DIR",
      help="if set, will copy the download or created databases to the given path location.",
      )
  parser.add_option("-V", "--version", action="store", dest="version",
      default="x.y.z", metavar="VERSION",
      help="if it makes sense, choose a version name that will be used to mark the project documentation, otherwise, leave it unassigned (defaults to '%default')"
      )
  parser.add_option("--dry-run", action="store_true", dest="dryrun",
      default=False, help="If set, doesn't execute the action, just test."
      )
  parser.add_option("-T", "--tests-regex", action="store", dest="tregex",
      default="", metavar="REGEXP",
      help="Filter tests to be executed by setting this option with a regular expression matching the test or tests you want to execute. This option is only in action if the action 'test' is used. (defaults to '%default')"
      )
  parser.add_option("-v", "--verbose", action="callback",
      callback=adm.build.increase_verbosity,
      help="increases the current verbosity level",
      )
  
  options, args = parser.parse_args()

  #some error checking
  if options.action not in actions:
    parser.error("option --action has to be one of %s" % ", ".join(actions))
  if options.build_type not in build_types:
    parser.error("option --build-type has to be one of %s" % ", ".join(build_types))
  if options.build_block not in build_blocks:
    parser.error("option --build-block has to be one of %s" % ", ".join(build_blocks))
  if options.jobs < 1:
    parser.error("option --jobs has to be equal or greater than 1")
  if options.debug_build and options.jobs > 1:
    logging.info("option --jobs will be reset to 1")
    options.jobs = 1
  if args:
    parser.error("this program does not accept positional arguments: %s" % args)

  options.platform = adm.build.platform(options)
  options.source_dir = sources

  #we also replace potential %(bla)s substitutions we may have
  options.version = options.version.strip()
  options.version = adm.build.untemplatize_version(options.version, options)
  options.build_prefix = options.build_prefix.strip()
  options.build_prefix = adm.build.untemplatize_path(options.build_prefix,
      options)
  options.install_prefix = options.install_prefix.strip()
  options.install_prefix = adm.build.untemplatize_path(options.install_prefix,
      options)
  options.doc_prefix = options.doc_prefix.strip()
  options.doc_prefix = adm.build.untemplatize_path(options.doc_prefix, options)
  if options.db_prefix:
    options.db_prefix = options.db_prefix.strip()
    options.db_prefix = adm.build.untemplatize_path(options.db_prefix, options)

  #some fixing
  if options.install_prefix and options.install_prefix[0] != os.path.sep:
    options.install_prefix = os.path.join(pwd, options.install_prefix)
  if options.build_prefix and options.build_prefix[0] != os.path.sep:
    options.build_prefix = os.path.join(pwd, options.build_prefix) 
  if options.doc_prefix and options.doc_prefix[0] != os.path.sep:
    options.doc_prefix = os.path.join(pwd, options.doc_prefix)

  return options, args

if __name__ == '__main__':
  (options, args) = parse_args()

  if options.action == 'build':
    #special option to run cmake/make_install
    adm.build.cmake(options)
    adm.build.install(options)
  elif options.action == 'cmake': adm.build.cmake(options)
  elif options.action == 'dot': adm.build.dot(options)
  elif options.action in ('make_all', 'all'): adm.build.make(options, 'all')
  elif options.action in ('make_install', 'install'): adm.build.install(options)
  elif options.action in ('documentation', 'docs'): adm.build.documentation(options)
  elif options.action == 'doxygen': adm.build.doxygen(options)
  elif options.action == 'sphinx': adm.build.sphinx(options)
  elif options.action in ('ctest', 'test'): adm.build.ctest(options)
  elif options.action in ('make_clean', 'clean'): adm.build.make(options, 'clean')
  elif options.action == 'mrproper': adm.build.mrproper(options)
  elif options.action == 'group_write': adm.build.group_write(options)
  elif options.action == 'group_unwrite': adm.build.group_unwrite(options)

  sys.exit(0)
