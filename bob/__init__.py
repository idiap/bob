# see https://docs.python.org/3/library/pkgutil.html
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)


def get_config():
  """
  Returns a string containing the configuration information.
  """
  import bob.extension
  return bob.extension.get_config(__name__)

  

