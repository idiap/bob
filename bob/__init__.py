#see http://peak.telecommunity.com/DevCenter/setuptools#namespace-packages
__import__('pkg_resources').declare_namespace(__name__)


import os
import pkgtools.pypi
import distutils.version
import pkg_resources

def get_url(package_name):
  "Given a package name get, from PyPI, the URL name"
  return pkgtools.pypi.PyPIJson(package_name).retrieve()['urls'][0]['url']

def get_config():
  """Returns a string containing the configuration information.
  """
  import bob.extension
  return bob.extension.get_config(__name__)

  
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
    
def get_dependencies(pkg_name="bob"):
  package      = pkg_resources.working_set.by_key[pkg_name]
  return [str(r) for r in package.requires()]
  
  
