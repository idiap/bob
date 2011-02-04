#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 08 Jul 2010 09:18:28 CEST 

import sys, os

# Imports our admin toolkit
install_dir = os.path.realpath(os.path.dirname(sys.argv[0]))
sys.path.append(install_dir)
import adm

if __name__ == '__main__':

  options, arguments = adm.environment.parse_args()
  if not arguments: arguments = [os.environ['SHELL']]

  if options.verbose:
    print "Starting '%s' inside torch5spro (%s/%s) environment" % \
      (os.path.basename(arguments[0]), options.version, options.arch)
  
  new_environ = adm.environment.generate_environment(options)
  if options.verbose >= 2: #print changed items
    for key, value in new_environ.iteritems():
      if os.environ.has_key(key) and os.environ[key] != value:
        print "Key:", key
        print "-", os.environ[key]
        print "+", new_environ[key]
      elif not os.environ.has_key(key):
        print "Key:", key
        print "=", new_environ[key]

  # The next line will add options to the program if torch can, to customize
  # the program behavior to torch environment specificities. This will be done
  # unless the user says explicitely that it does not want prompt fiddling.
  if options.env_manipulation:
    adm.environment.set_prompt(arguments, new_environ)

  if options.verbose >= 2: print "Executing '%s'" % ' '.join(arguments)

  retval = os.spawnvpe(os.P_WAIT, arguments[0], arguments, new_environ)
  if options.verbose >= 1:
    print "Program '%s' exited with status %d" % (arguments[0], retval)
  sys.exit(retval)
