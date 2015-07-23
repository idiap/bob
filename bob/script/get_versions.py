# Andre Anjos <andre.anjos@idiap.ch>
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Mon 20 Jul 17:30:00 CEST 2015

# Lists the final version of a given package in PyPI
# Uses the package 'pkgtools' for such.

import os
import sys
import re
import pkgtools.pypi
import distutils.version
import pkg_resources

PKG_NAME = "bob"

def get_releases(package):
  try:
    return pkgtools.pypi.PyPIJson(package).retrieve()['releases'].keys()
  except:
    return []

def get_max_version(versions):

  try:
    v = list(reversed(sorted([distutils.version.StrictVersion(k) for k in versions])))
    final = [k for k in v if not k.prerelease]
    if final: return final[0]
    return v[0]
  except:
    v = list(reversed(sorted([distutils.version.LooseVersion(k) for k in versions])))
    final = [k for k in v if not re.search(r'[a-z]', k.vstring)]
    if final: return final[0]
    return v[0]
    
def get_dependencies():
  package      = pkg_resources.working_set.by_key[PKG_NAME]
  return [str(r) for r in package.requires()]

def main():
  
  dependencies = get_dependencies()
  for i in range(2,len(dependencies)):
    d = dependencies[i].split("==")[0]
    versions = get_releases(d)
    print "{0} == {1}".format(d,get_max_version(versions))

if __name__ == '__main__':
  main()
