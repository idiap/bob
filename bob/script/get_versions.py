# Andre Anjos <andre.anjos@idiap.ch>
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Mon 20 Jul 17:30:00 CEST 2015

# Lists the final version of a given package in PyPI
# Uses the package 'pkgtools' for such.

import sys
import os
def main():

  try:
    from bob.utils import get_dependencies, get_releases, get_max_version

    if len(sys.argv) != 2:
      print "usage: %s <package>" % os.path.basename(sys.argv[0])
      sys.exit(1)  

    dependencies = get_dependencies(pkg_name = sys.argv[1])
    for i in range(2,len(dependencies)):
      d = dependencies[i].split("==")[0]
      versions = get_releases(d)
      print "{0} == {1}".format(d,get_max_version(versions))
  except ImportError:
    print("Package pkgtools required, please install it.  <https://pypi.python.org/pypi/pkgtools/>")

if __name__ == '__main__':
  main()
