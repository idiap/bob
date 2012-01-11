#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 08 Jul 2010 09:18:28 CEST 

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
