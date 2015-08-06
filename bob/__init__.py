#see http://peak.telecommunity.com/DevCenter/setuptools#namespace-packages
__import__('pkg_resources').declare_namespace(__name__)


def get_config():
  """
  Returns a string containing the configuration information.
  """
  import bob.extension
  return bob.extension.get_config(__name__)

  

