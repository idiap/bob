#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 08 Jul 2010 09:18:28 CEST 

import sys, os, subprocess

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
    print "Here are the environment *modifications*:"
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

  try:
    p = subprocess.Popen(arguments, env=new_environ)
  except OSError, e:
    # occurs when the file is not executable or not found
    print "Error executing '%s': %s (%d)" % (' '.join(arguments), e.strerror,
        e.errno)
    sys.exit(e.errno)

  (stdout, stderr) = p.communicate()
  if options.verbose >= 1:
    if p.returncode != 0: print "Error:",
    print "Program '%s' exited with status %d" % (' '.join(arguments), p.returncode)
  sys.exit(p.returncode)
