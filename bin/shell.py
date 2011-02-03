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
  if not arguments: arguments = [os.environ['SHELL']]

  print "Starting '%s' inside torch5spro (%s/%s) environment" % \
    (os.path.basename(arguments[0]), options.version, options.arch)
  new_environ = adm.environment.generate_environment(options)
  if options.verbose: #print changed items
    for key, value in new_environ.iteritems():
      if os.environ.has_key(key) and os.environ[key] != value:
        print "Key:", key
        print "-", os.environ[key]
        print "+", new_environ[key]
      elif not os.environ.has_key(key):
        print "Key:", key
        print "=", new_environ[key]
  if options.verbose: print "Executing '%s'" % ' '.join(arguments)
  retval = os.spawnvpe(os.P_WAIT, arguments[0], arguments, new_environ)
  if options.verbose:
    print "Program '%s' exited with status %d" % (arguments[0], retval)
  sys.exit(retval)
