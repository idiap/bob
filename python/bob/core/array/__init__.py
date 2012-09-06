from .. import convert as __private_convert__
import warnings

def convert(*args, **kwargs):
  warnings.warn("bob.core.array.convert() has moved to bob.core.convert()", DeprecationWarning)
  return __private_convert__(*args, **kwargs)
convert.__doc__ = __private_convert__.__doc__

__all__ = ['convert']
