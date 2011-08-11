#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 12 Aug 2010 13:16:09 CEST 

"""Tools for building torch.
"""

import os
import sys
import subprocess
import logging
import tempfile
import time
import pprint
import fnmatch
import shutil

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
  """Increases the current verbosity level for the logging module. 
  
  This method can be used as a callback for optparse.
  """
  global CURRENT_LOGGING_LEVEL
  next = CURRENT_LOGGING_LEVEL - 1
  if next < 0: next = 0
  logger = logging.getLogger()
  logger.setLevel(LOGGING_LEVELS[next])
  CURRENT_LOGGING_LEVEL = next

def decrease_verbosity(option, opt, value, parser):
  """Decreases the current verbosity level for the logging module.
  
  This method can be used as a callback for optparse.
  """
  global CURRENT_LOGGING_LEVEL
  next = CURRENT_LOGGING_LEVEL + 1
  if next > (len(LOGGING_LEVELS)-1): next = len(LOGGING_LEVELS) - 1
  logger = logging.getLogger()
  logger.setLevel(LOGGING_LEVELS[next])
  CURRENT_LOGGING_LEVEL = next

def run(cmd, log=False, dir=None, prefix=None):
  """Executes command 'cmd' on the shell. If 'log' is set, save the output on
  'dir/prefix.txt'.

  Returns the output status after executing 'cmd'.
  """

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

def cmake(option):
  """Builds the project using cmake at 'option.build_prefix', install it at
  'option.install_prefix'.
  
  If there is a problem, throw a RuntimeError.
  """
  if (option.dryrun):
    logging.info("Dry run: action 'cmake' will not be executed")
    return

  logging.debug('Running cmake...')
  
  if os.path.exists(option.build_prefix) and hasattr(option, "cleanup") and \
      option.cleanup:
    logging.debug('Removing build directory %s before build on user request' % option.build_prefix)
    shutil.rmtree(option.build_prefix)

  if os.path.exists(option.install_prefix) and hasattr(option, "cleanup") and \
      option.cleanup:
    logging.debug('Removing install directory %s before build on user request' % option.install_prefix)
    shutil.rmtree(option.install_prefix)

  if not os.path.exists(option.build_prefix): os.makedirs(option.build_prefix)

  os.chdir(option.build_prefix)

  cmake_options = {}
  if option.graphviz: cmake_options['--graphviz'] = "dependencies.dot"
  cmake_options['-DCMAKE_BUILD_TYPE'] = option.build_type
  cmake_options['-DCMAKE_INSTALL_PREFIX'] = option.install_prefix
  cmake_options['-DTORCH_LINKAGE'] = 'dynamic'
  if option.static_linkage: cmake_options['-DTORCH_LINKAGE'] = 'static'
  
  if option.build_block == 'all':
    cmake_options['-DTORCH_CXX'] = 'ON'
    cmake_options['-DTORCH_PYTHON'] = 'ON'
  else: 
    cmake_options['-DTORCH_%s' % option.build_block.upper()] = 'ON'
  
  if option.createdb:
    cmake_options['-DTORCH_CREATE_DATABASES'] = 'ON'
  else:
    cmake_options['-DTORCH_CREATE_DATABASES'] = 'OFF'

  if option.db_prefix:
    cmake_options['-DTORCH_DB_INSTALL_PREFIX'] = option.db_prefix
  else:
    cmake_options['-DTORCH_DB_INSTALL_PREFIX'] = 'OFF'

  if option.createdb and option.db_prefix and os.path.exists(option.db_prefix):
    # touch it so to modify the last modification time
    os.utime(option.db_prefix, None)

  cmdline = ['cmake']
  if option.debug_build: cmdline.append('--debug-output')
  
  for k,v in cmake_options.items(): cmdline.append('%s=%s' % (k, v))
  cmdline.append(option.source_dir)
  
  if hasattr(option, "log_prefix"):
    status = run(cmdline, option.save_output, option.log_prefix, cmdline[0])
  else:
    status = run(cmdline)
  if status != 0:
    raise RuntimeError('** ERROR: "cmake" did not work as expected.')
  
  logging.debug('Finished running cmake.')

def install(option):
  """Runs "make install" on the 'option.build_prefix'. If there is a problem, 
  throws RuntimeError.
  """
  if (option.dryrun):
    logging.info("Dry run: action 'install' will not be executed")
    return

  import shutil

  logging.debug('Running make install...')
  make(option, 'install')

  logging.debug('Installing setup scripts...')

  # copies all relevant files setup, if they don't already exist
  srcdir = os.path.realpath(os.path.join(option.source_dir, '..'))
  destdir = os.path.realpath(os.path.join(option.install_prefix, '..', '..'))

  # we go through extra copying if installing outside the source tree
  if srcdir != destdir:
    bindestdir = os.path.join(destdir, 'bin')
    if os.path.exists(bindestdir): shutil.rmtree(bindestdir)
    shutil.copytree(os.path.join(srcdir, 'bin'), bindestdir)

    # copies a reference of the sources used for the build
    srcdestdir = os.path.join(destdir, 'src')
    if os.path.exists(srcdestdir): shutil.rmtree(srcdestdir)
    shutil.copytree(os.path.join(srcdir, 'src'), srcdestdir)

    # finally, writes a ".version" file at the root of the directory
    info = os.path.join(destdir, ".version")
    if os.path.exists(info): os.unlink(info)
    f = open(info, 'wt')
    f.write(option.version)
    f.close()

def make(option, target="all"):
  """Runs "make 'target'" on the 'option.build_prefix'. If there is a problem, 
  throws RuntimeError.
  """

  if (option.dryrun):
    logging.info("Dry run: action 'make' for target '%s' will not be executed" % target)
    return

  logging.debug('Running make %s...' % target)

  os.chdir(option.build_prefix)

  cmdline = ['make', '--keep-going']
  if option.debug_build:
    cmdline.append('VERBOSE=1')
  else:
    cmdline.append('-j%d' % option.jobs)
  cmdline.append(target)
  if hasattr(option, "log_prefix"):
    status = run(cmdline, option.save_output, option.log_prefix, cmdline[0]+'_'+target)
  else:
    status = run(cmdline)
  if status != 0:
    raise RuntimeError('** ERROR: "make %s" did not work as expected.' % target)
  logging.debug('Finished running make %s.' % target)

def ctest(option):
  """Runs ctest on the 'option.build_prefix'. If there is a problem, 
  throws RuntimeError.
  """

  if (option.dryrun):
    logging.info("Dry run: action 'ctest' will not be executed")
    return

  logging.debug('Running ctest...')

  os.chdir(option.build_prefix)

  cmdline = ['ctest']
  if option.debug_build:
    cmdline.append('--verbose')
  else:
    cmdline.append('-j%d' % option.jobs)
  if option.tregex:
    cmdline.extend(('--tests-regex', option.tregex))
  if hasattr(option, "log_prefix"):
    status = run(cmdline, option.save_output, option.log_prefix, 'make_test')
  else:
    status = run(cmdline)
  if status != 0:
    raise RuntimeError('** ERROR: "ctest" did not work as expected.')
  logging.debug('Finished running ctest.')

def documentation(option):
  """Builds the project documentation using doxygen and sphinx. If there is
  a problem, throws a RuntimeError."""
  
  if (option.dryrun):
    logging.info("Dry run: action 'documentation' will not be executed")
    return

  logging.debug('Building documentation...')

  if os.path.exists(option.doc_prefix) and hasattr(option, "cleanup") and \
      option.cleanup:
    logging.debug('Removing directory %s before doxygen on user request' % \
        option.doc_prefix)
    shutil.rmtree(option.doc_prefix)

  if not os.path.exists(option.doc_prefix): os.makedirs(option.doc_prefix)

  doxygen(option)
  sphinx(option)

  logging.debug('Finished building documentation.')

def doxygen(option):
  """Builds the project documentation using doxygen. If there is a problem,
  throws a RuntimeError."""

  if (option.dryrun):
    logging.info("Dry run: action 'doxygen' will not be executed")
    return

  logging.debug('Running doxygen...')

  doxygen_prefix = os.path.join(option.doc_prefix, "doxygen")
  if not os.path.exists(doxygen_prefix): os.makedirs(doxygen_prefix)
  
  overwrite_options = {}
  overwrite_options['PROJECT_NUMBER'] = option.version
  overwrite_options['INPUT'] = option.source_dir
  overwrite_options['STRIP_FROM_PATH'] = option.source_dir
  overwrite_options['OUTPUT_DIRECTORY'] = doxygen_prefix
  if option.debug_build: overwrite_options['QUIET'] = 'NO'
  extras = []
  for k,v in overwrite_options.items(): extras.append('%s = %s\n' % (k, v))

  original = file(option.doxyfile, 'rt')
  lines = original.readlines() + extras
  original.close()
  (tmpfd, tmpname) = tempfile.mkstemp()
  tmpfile = os.fdopen(tmpfd, 'wt')
  tmpfile.writelines(lines)
  tmpfile.seek(0)
 
  cmdline = ['doxygen', tmpname]
  if hasattr(option, "log_prefix"):
    status = run(cmdline, option.save_output, option.log_prefix, cmdline[0])
  else:
    status = run(cmdline)
  if status != 0:
    raise RuntimeError('** ERROR: "doxygen" did not work as expected.')
  tmpfile.close()
  os.unlink(tmpname)

  #create a link from index.html to main.html 
  os.chdir(os.path.join(option.doc_prefix, 'doxygen', 'html'))
  if not os.path.exists('main.html'):
    logging.debug("Copying index.html -> main.html")
    shutil.copy('index.html', 'main.html')

  logging.debug('Finished running doxygen.')

def sphinx(option):
  """Builds the project user guide using sphinx. If there is a problem,
  throws a RuntimeError."""

  if (option.dryrun):
    logging.info("Dry run: action 'sphinx' will not be executed")
    return

  logging.debug('Running Sphinx...')

  sphinx_prefix = os.path.join(option.doc_prefix, "sphinx")
  if not os.path.exists(sphinx_prefix): os.makedirs(sphinx_prefix)
  
  overwrite_options = {}
  overwrite_options['PROJECT_NUMBER'] = option.version
  overwrite_options['INPUT'] = option.source_dir
  overwrite_options['STRIP_FROM_PATH'] = option.source_dir
  overwrite_options['OUTPUT_DIRECTORY'] = sphinx_prefix
  
  # Look for sphinx-build executable
  #cmdline = ['sphinx-build']
  try:
    py_version = '.'.join(str(n) for n in sys.version_info[0:2])
    sphinx_bin = 'sphinx-build-' + py_version
    cmdline = [sphinx_bin]
    fnull = open(os.devnull, 'w') 
    p = subprocess.Popen(cmdline, stdin=None, stdout=fnull, stderr=fnull)
    p.wait()
    fnull.close()
  except OSError:
    try:
      cmdline = ['sphinx-build']
      fnull = open(os.devnull, 'w') 
      p = subprocess.Popen(cmdline, stdin=None, stdout=fnull, stderr=fnull)
      p.wait()
      fnull.close()
    except OSError:
      raise RuntimeError('** ERROR: Not able to find "sphinx-build" executable.')
   
  cmdline.extend(['-c', option.sphinxdir])
  # TODO: Parse version
  cmdline.extend(['-D','version=%s' % option.version])
  cmdline.extend(['-D','release=%s' % option.version])
 
  # Generates HTML
  sphinx_prefix_html = os.path.join(sphinx_prefix, "html")
  cmdline_html = cmdline[:]
  cmdline_html.extend(['-b', 'html', option.source_dir, sphinx_prefix_html])
  if not os.path.exists(sphinx_prefix_html): os.makedirs(sphinx_prefix_html)
  if hasattr(option, "log_prefix"):
    status = run(cmdline_html, option.save_output, option.log_prefix)
  else:
    status = run(cmdline_html)
  if status != 0:
    raise RuntimeError('** ERROR: "sphinx-build -b html" did not work as expected.')

  # Generates the LaTeX sources (not the final PDF)
  sphinx_prefix_latex = os.path.join(sphinx_prefix, "latex")
  cmdline_latex = cmdline[:]
  cmdline_latex.extend(['-b', 'latex', option.source_dir, sphinx_prefix_latex])
  if not os.path.exists(sphinx_prefix_latex): os.makedirs(sphinx_prefix_latex)
  if hasattr(option, "log_prefix"):
    status = run(cmdline_latex, option.save_output, option.log_prefix)
  else:
    status = run(cmdline_latex)
  if status != 0:
    raise RuntimeError('** ERROR: "sphinx-build -b latex" did not work as expected.')

  # Runs the coverage tests
  sphinx_prefix_latex = os.path.join(sphinx_prefix, "coverage")
  cmdline_latex = cmdline[:]
  cmdline_latex.extend(['-b', 'coverage', option.source_dir, sphinx_prefix_latex])
  if not os.path.exists(sphinx_prefix_latex): os.makedirs(sphinx_prefix_latex)
  if hasattr(option, "log_prefix"):
    status = run(cmdline_latex, option.save_output, option.log_prefix)
  else:
    status = run(cmdline_latex)
  if status != 0:
    raise RuntimeError('** ERROR: "sphinx-build -b coverage" did not work as expected.')

  logging.debug('Finished running Sphinx.')

def differences(option):
  """Calculates the repository differences since the time specified."""
  
  if (option.dryrun):
    logging.info("Dry run: action 'differences' will not be executed")
    return

  logging.debug('Running git diff...')

  try:
    start = time.localtime(int(option.diffs_since))
    start = time.strftime('%d.%m.%Y %H:%M:%S', start)
  except ValueError:
    start = option.diffs_since

  cmd=['git']
  cmd.append('--git-dir=%s' % option.repository)
  cmd.append('log')
  cmd.append('--since="%s"' % start)

  if hasattr(option, "log_prefix"):
    status = run(cmd, option.save_output, option.log_prefix, 'differences')
  else:
    status = run(cmd)
  if status != 0:
    raise RuntimeError('** ERROR: "git-log" did not work as expected.')

  logging.debug('Finished running git diff.')

def status_log(option, timing, problems):
  """Writes a pythonic status file in the root of the log directory. 
  
  This file can be later imported by an analysis script to find out the nightly
  build results.
  """

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
  lines.append('# calculated automatically by %s' % prog)
  lines.append('platform = \'%s\'' % platform(option))
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

def dot(option):
  """Runs dot on the output of cmake/graphviz. Raises a RuntimeError if there
  is any problem."""

  if (option.dryrun):
    logging.info("Dry run: action 'dot' will not be executed")
    return

  logging.debug('Running dot...')

  os.chdir(option.build_prefix)
  dotfile = 'dependencies.dot'
  if hasattr(option, "save_output") and option.save_output: 
    os.chdir(option.log_prefix)
    dotfile = os.path.join('..', dotfile)

  cmdline = ['dot', '-Tpng', dotfile, '-odependencies.png']
  if hasattr(option, "log_prefix"):
    status = run(cmdline, option.save_output, option.log_prefix, cmdline[0])
  else:
    status = run(cmdline)
  if status != 0:
    raise RuntimeError('** ERROR: "dot" did not work as expected.')
  logging.debug('Finished running dot.')

def platform(option):
  """Calculates the platform string."""
  import platform
  
  base = platform.system().lower()
  if base == 'darwin': 
    base = 'macosx' #change name to something nicer and easy to identify

  arch = platform.architecture()[0]
  if arch == '32bit': arch = 'i686'
  elif arch == '64bit': arch = 'x86_64'

  if hasattr(option, "force_32bits") and option.force_32bits: 
    logging.warn("Forcing 32-bits compilation")
    arch = 'i686'

  return '%s-%s-%s' % (base, arch, option.build_type)

def action(what, option, *args):
  start = time.time()
  problems = ('success',)
  try:
    what(option, *args)
  except Exception as e:
    logging.error('Executing action "%s": %s' % (what.__name__, e))
    problems = ('failed', '%s' % e)
  total_time = time.time() - start 
  logging.info('Action "%s" took %.1f s' % (what.__name__, total_time))
  return (total_time, problems)

def mrproper(option):
  """Completely sanitizes the build and installation areas"""

  if (option.dryrun):
    logging.info("Dry run: action 'mrproper' will not be executed")
    return

  import shutil
  shutil.rmtree(option.install_prefix, ignore_errors=True)
  shutil.rmtree(option.build_prefix, ignore_errors=True)
  shutil.rmtree(option.doc_prefix, ignore_errors=True)
  p = subprocess.Popen(['find', '.', '-name', '*~', '-or', '-iname', '*.pyc'], stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT)
  (out, err) = p.communicate()
  for f in [k.strip() for k in out.split('\n') if k.strip()]:
    os.unlink(os.path.realpath(f))

def untemplatize_path(path, option):
  """Removes string templates that may have been inserted into the path
  descriptor and returns a fully resolved string.
  """
  replacements = {
      'name': 'torch5spro',
      'version': option.version,
      'date': time.strftime("%d.%m.%Y"),
      'weekday': time.strftime("%A").lower(),
      'platform': option.platform,
      'install-prefix': option.install_prefix,
      'build-prefix': option.build_prefix,
      'doc-prefix': option.doc_prefix,
      }
  retval = path % replacements
  if retval.find('%(') != -1:
    raise RuntimeError("Cannot fully expand path `%s'" % retval)
  return retval

def untemplatize_version(version, option):
  replacements = {
      'name': 'torch5spro',
      'date': time.strftime("%d.%m.%Y"),
      'weekday': time.strftime("%A").lower(),
      'platform': option.platform,
      'install-prefix': option.install_prefix,
      'build-prefix': option.build_prefix,
      'doc-prefix': option.doc_prefix,
      }
  retval = version % replacements
  if retval.find('%(') != -1:
    raise RuntimeError("Cannot fully expand version `%s'" % retval)
  return retval
