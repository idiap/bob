#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# André Anjos <andre.anjos@idiap.ch>
# Fri Aug 6 17:57:15 2010 +0200
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

import sys, os

# Imports our admin toolkit
sys.path.append(os.path.realpath(os.path.dirname(sys.argv[0])))
import adm

if __name__ == '__main__':

  options, arguments = adm.environment.parse_args()
  new_environ = adm.environment.generate_environment(options)

  #echo what will be setup
  setupline = 'Setting-up current shell for bob (%s/%s)'
  print adm.environment.shell_echo(setupline % (options.version, options.arch))

  for key, value in new_environ.iteritems():
    if os.environ.has_key(key) and os.environ[key] != value:
      print adm.environment.shell_str(key, value, options.csh)
    elif not os.environ.has_key(key):
      print adm.environment.shell_str(key, value, options.csh)

  sys.exit(0)
