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

def get_headers(dir, excludes):
  """Gets all files ending in '.h' from the directory, recursively, except for
  what is defined in the input argument "excludes"."""
  retval = []
  for (path, dirs, files) in os.walk(dir):
    for f in fnmatch.filter(files, '*.h'):
      if f in excludes: continue
      sub = path.replace(dir+os.sep, '')
      retval.append(os.path.join(sub, f))
  return retval

def write_header(option):
  """Writes a new header file that incorporates all existing ones."""
  scandir = os.path.join(option.install_prefix, 'include', 'torch')
  output = os.path.join(scandir, 'torch5spro.h')
  excludes = [
              os.path.basename(output),
              'THTensorGen.h', 
              'THStorageGen.h', 
              'TensorGen.h',
             ]
  headers = get_headers(scandir, excludes)
  f = open(output, 'wt')
  f.write('/* This file was automatically generated -- DO NOT CHANGE IT */\n')
  f.write('/* Date: %s */\n\n' % time.asctime())
  f.write('#ifndef __TORCH5SPROC_H__\n')
  f.write('#define __TORCH5SPROC_H__\n\n')
  f.writelines(['#include "%s"\n' % k for k in headers])
  f.write('\n#endif /* __TORCH5SPROC_H__ */\n')
  f.close()

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

def run(cmd, log, dir, prefix):
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
  logging.debug('Running cmake...')
  
  if os.path.exists(option.build_prefix) and option.cleanup:
    logging.debug('Removing build directory %s before build on user request' % option.build_prefix)
    shutil.rmtree(option.build_prefix)

  if os.path.exists(option.install_prefix) and option.cleanup:
    logging.debug('Removing install directory %s before build on user request' % option.install_prefix)
    shutil.rmtree(option.install_prefix)

  if not os.path.exists(option.build_prefix): os.makedirs(option.build_prefix)

  os.chdir(option.build_prefix)

  cmake_options = {}
  cmake_options['--graphviz'] = "dependencies.dot"
  cmake_options['-DCMAKE_BUILD_TYPE'] = option.build_type
  cmake_options['-DCMAKE_INSTALL_PREFIX'] = option.install_prefix
  cmake_options['-DINCLUDE_DIR'] = \
      os.path.join(option.install_prefix, 'include')
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

def make(option, target="all"):
  """Runs "make 'target'" on the 'option.build_prefix'. If there is a problem, 
  throws RuntimeError.
  """

  logging.debug('Running make %s...' % target)

  os.chdir(option.build_prefix)

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

def ctest(option):
  """Runs ctest on the 'option.build_prefix'. If there is a problem, 
  throws RuntimeError.
  """

  logging.debug('Running ctest...')

  os.chdir(option.build_prefix)

  cmdline = ['ctest']
  if option.debug_build:
    cmdline.append('--verbose')
  status = run(cmdline, option.save_output, option.log_prefix, 'make_test')
  if status != 0:
    raise RuntimeError, '** ERROR: "ctest" did not work as expected.'
  logging.debug('Finished running ctest.')

def doxygen(option):
  """Builds the project documentation using doxygen. If there is a problem,
  throws a RuntimeError."""

  logging.debug('Running doxygen...')

  if os.path.exists(option.doc_prefix) and option.cleanup:
    logging.debug('Removing directory %s before doxygen on user request' % \
        option.doc_prefix)
    shutil.rmtree(option.doc_prefix)

  if not os.path.exists(option.doc_prefix): os.makedirs(option.doc_prefix)

  overwrite_options = {}
  overwrite_options['PROJECT_NUMBER'] = option.version
  overwrite_options['INPUT'] = option.source_dir
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

def differences(option):
  """Calculates the repository differences since the time specified."""
  
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

  status = run(cmd, option.save_output, option.log_prefix, 'differences')
  if status != 0:
    raise RuntimeError, '** ERROR: "git-log" did not work as expected.'

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

  logging.debug('Running dot...')

  os.chdir(option.build_prefix)
  dotfile = 'dependencies.dot'
  if option.save_output: 
    os.chdir(option.log_prefix)
    dotfile = os.path.join('..', dotfile)

  cmdline = ['dot', '-Tpng', dotfile, '-odependencies.png']
  status = run(cmdline, option.save_output, option.log_prefix, cmdline[0])
  if status != 0:
    raise RuntimeError, '** ERROR: "dot" did not work as expected.'
  logging.debug('Finished running dot.')

def platform(option):
  """Calculates the platform string."""
  uname = os.uname()
  return '%s-%s-%s' % (uname[0].lower(), uname[4].lower(), option.build_type)

def action(what, option, *args):
  start = time.time()
  problems = ('success',)
  try:
    what(option, *args)
  except Exception, e:
    logging.error('Executing action "%s": %s' % (what.__name__, e))
    problems = ('failed', '%s' % e)
  total_time = time.time() - start 
  logging.info('Action "%s" took %.1f s' % (what.__name__, total_time))
  return (total_time, problems)
