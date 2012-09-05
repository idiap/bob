#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 15 Feb 2012 12:36:00 CET 

"""A few helpers for testing and other.
"""

def unittest_main(cls):
  """This method helps users writing unit test scripts in creating a main
  function that can automatically take into consideration verbosity settings
  and correctly handles data directory placement."""

  class method(object):
    """Main function for tests"""

    def __init__(self, cls):
      self.cls = cls #main test class

    def __call__(self):
      import argparse
      import os
      import sys
      import unittest

      parser = argparse.ArgumentParser(description=__doc__)
      parser.add_argument("--verbose", "-v", action="count", default=0,
          help="Run verbosity")
      parser.add_argument("--cwd", "-c", help="Change the working directory to this subdirectory before starting the tests")
      args = parser.parse_args()

      if args.cwd:
        cwd = os.path.realpath(args.cwd)
        if os.path.exists(cwd): 
          if args.verbose > 0: print "Changing directory to '%s'..." % (cwd,),
          os.chdir(cwd)
          if args.verbose > 0: print "done."
        else:
          raise RuntimeError, "Test module set to execute from data directory, but the path '%s' does not point to a valid directory. Please check." % (cwd,)

      suite = unittest.TestLoader().loadTestsFromTestCase(self.cls)
      results = unittest.TextTestRunner(verbosity=args.verbose).run(suite)

      # exit with posix compliant status
      if results.wasSuccessful(): sys.exit(0)
      else: sys.exit(1)

  return method(cls)
