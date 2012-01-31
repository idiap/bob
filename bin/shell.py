#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# André Anjos <andre.anjos@idiap.ch>
# Thu Jan 20 15:28:59 2011 +0100
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

import sys, os, subprocess
import shlex

# Imports our admin toolkit.
if os.environ.has_key('OVERWRITE_BOB5SPRO_BINROOT'):
  # grid environments will use this variant:
  install_dir = os.environ['OVERWRITE_BOB5SPRO_BINROOT']
else:
  # normal execution path:
  install_dir = os.path.realpath(os.path.dirname(sys.argv[0]))
sys.path.append(install_dir)
import adm

if __name__ == '__main__':
  
  # A little help if we are called in shebang mode
  args = sys.argv
  if len(sys.argv) > 1 and sys.argv[1].find(' ') != -1:
    # Assuming this is a shebang call, only use 1st argument
    args = [sys.argv[0]] + shlex.split(sys.argv[1]) + sys.argv[2:]

  options, arguments = adm.environment.parse_args(args)
  if not arguments: arguments = [os.environ['SHELL']]

  if options.verbose:
    print("Starting '%s' inside bob (%s/%s) environment" % \
      (os.path.basename(arguments[0]), options.version, options.arch))
  
  new_environ = adm.environment.generate_environment(options)
  if options.verbose >= 2: #print changed items
    print("Here are the environment *modifications*:")
    for key, value in new_environ.items():
      if key in os.environ and os.environ[key] != value:
        print("Key:", key)
        print("-", os.environ[key])
        print("+", new_environ[key])
      elif key not in os.environ:
        print("Key:", key)
        print("=", new_environ[key])

  # The next line will add options to the program if bob can, to customize
  # the program behavior to bob environment specificities. This will be done
  # unless the user says explicitely that it does not want prompt fiddling.
  if options.env_manipulation:
    adm.environment.set_prompt(arguments, new_environ)

  if options.verbose >= 2: print("Executing '%s'" % ' '.join(arguments))

  try:
    p = subprocess.Popen(arguments, env=new_environ)
  except OSError as e:
    # occurs when the file is not executable or not found
    print("Error executing '%s': %s (%d)" % (' '.join(arguments), e.strerror,
        e.errno))
    sys.exit(e.errno)
  
  try:
    p.communicate()
  except KeyboardInterrupt: # the user CTRL-C'ed
    if options.verbose >= 1:
      print("User interrupt, killing '%s'..." % (' '.join(arguments)))
    import signal
    os.kill(p.pid, signal.SIGTERM)
    sys.exit(signal.SIGTERM)
  if options.verbose >= 1:
    if p.returncode != 0:
      print("Error: Program '%s' exited with status %d" % (' '.join(arguments), p.returncode))
  sys.exit(p.returncode)
